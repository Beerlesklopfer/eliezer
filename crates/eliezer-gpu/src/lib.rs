//! Herstellerunabhängige VRAM-Messung und Rückfall auf Buchführung.
use eliezer_core::config::{GpuVendor, ProbeMethod};
use std::{path::{Path,PathBuf}, sync::{Mutex,atomic::{AtomicU64,Ordering}}, process::{Command,Stdio}, io::Read, time::{Duration,Instant}};
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct GpuMemory {pub total_bytes:u64,pub used_bytes:u64}
impl GpuMemory {
    pub fn new(total_bytes:u64,used_bytes:u64)->Result<Self,ProbeError>{if used_bytes>total_bytes{Err(ProbeError::Inconsistent)}else{Ok(Self{total_bytes,used_bytes})}}
    pub fn free_bytes(self)->u64{self.total_bytes-self.used_bytes}
}
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum ProbeError{Unavailable,InvalidOutput,Inconsistent,Timeout}
pub trait GpuProbe:Send+Sync{fn memory(&self)->Result<GpuMemory,ProbeError>;fn name(&self)->&str;}
pub struct Accounting{pub budget:u64,pub leases:AtomicU64}
impl GpuProbe for Accounting{
    fn memory(&self)->Result<GpuMemory,ProbeError>{GpuMemory::new(self.budget,self.leases.load(Ordering::Relaxed))}
    fn name(&self)->&str{"none"}
}
pub struct Sysfs{pub root:PathBuf,pub index:u32}
impl GpuProbe for Sysfs{
    fn memory(&self)->Result<GpuMemory,ProbeError>{
        let root=self.root.join(format!("card{}/device",self.index));
        let read=|name|std::fs::read_to_string(root.join(name)).map_err(|_|ProbeError::Unavailable)?
            .trim().parse().map_err(|_|ProbeError::InvalidOutput);
        GpuMemory::new(read("mem_info_vram_total")?,read("mem_info_vram_used")?)
    }
    fn name(&self)->&str{"sysfs"}
}
pub fn detect_vendor(root:&Path,index:u32)->GpuVendor{
    match std::fs::read_to_string(root.join(format!("card{index}/device/vendor"))).ok().as_deref().map(str::trim){
        Some("0x1002")=>GpuVendor::Amd,Some("0x10de")=>GpuVendor::Nvidia,_=>GpuVendor::None,
    }
}
#[derive(Clone,Copy)]
pub enum SmiKind{Nvidia,Rocm}
pub struct Smi{pub program:PathBuf,pub index:u32,pub kind:SmiKind}
impl GpuProbe for Smi{
    fn memory(&self)->Result<GpuMemory,ProbeError>{
        let mut command=Command::new(&self.program);
        match self.kind{
            SmiKind::Nvidia=>{command.args(["--query-gpu=memory.total,memory.used","--format=csv,noheader,nounits",&format!("--id={}",self.index)]);},
            SmiKind::Rocm=>{command.args(["--showmeminfo","vram","--json","--device",&self.index.to_string()]);}
        }
        command.env_clear().env("PATH","/usr/bin:/bin").stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
        let mut child=command.spawn().map_err(|_|ProbeError::Unavailable)?;
        let stdout=child.stdout.take().ok_or(ProbeError::Unavailable)?;
        let (tx,rx)=std::sync::mpsc::sync_channel(1);
        std::thread::spawn(move||{let mut bytes=Vec::new();let result=stdout.take(65537).read_to_end(&mut bytes).map(|_|bytes);let _=tx.send(result);});
        let start=Instant::now();
        loop{
            match child.try_wait(){
                Ok(Some(status))=>{if !status.success(){return Err(ProbeError::Unavailable);}break;},
                Ok(None)=>{},Err(_)=>{let _=child.kill();let _=child.wait();return Err(ProbeError::Unavailable);}
            }
            if start.elapsed()>=Duration::from_millis(1900){let _=child.kill();let _=child.wait();return Err(ProbeError::Timeout);}
            std::thread::sleep(Duration::from_millis(5));
        }
        let remaining=Duration::from_millis(1900).saturating_sub(start.elapsed());
        let bytes=rx.recv_timeout(remaining).map_err(|_|ProbeError::Timeout)?.map_err(|_|ProbeError::InvalidOutput)?;
        if bytes.len()>65536{return Err(ProbeError::InvalidOutput);}
        let output=std::str::from_utf8(&bytes).map_err(|_|ProbeError::InvalidOutput)?;
        match self.kind{SmiKind::Nvidia=>parse_nvidia(output),SmiKind::Rocm=>parse_rocm(output,self.index)}
    }
    fn name(&self)->&str{match self.kind{SmiKind::Nvidia=>"nvidia-smi",SmiKind::Rocm=>"rocm-smi"}}
}
pub fn parse_nvidia(output:&str)->Result<GpuMemory,ProbeError>{
    let fields:Vec<_>=output.trim().split(',').collect();
    if fields.len()!=2{return Err(ProbeError::InvalidOutput);}
    let parse=|s:&str|s.trim().parse::<u64>().ok().and_then(|v|v.checked_mul(1_048_576)).ok_or(ProbeError::InvalidOutput);
    GpuMemory::new(parse(fields[0])?,parse(fields[1])?)
}
pub fn parse_rocm(output:&str,index:u32)->Result<GpuMemory,ProbeError>{
    let data:serde_json::Value=serde_json::from_str(output).map_err(|_|ProbeError::InvalidOutput)?;
    let card=&data[format!("card{index}")];
    let parse=|name:&str|card[name].as_str().and_then(|v|v.parse().ok()).or_else(||card[name].as_u64()).ok_or(ProbeError::InvalidOutput);
    GpuMemory::new(parse("VRAM Total Memory (B)")?,parse("VRAM Total Used Memory (B)")?)
}
pub struct Fallback{sources:Vec<Box<dyn GpuProbe>>,current:Mutex<usize>}
impl Fallback{
    pub fn new(sources:Vec<Box<dyn GpuProbe>>)->Self{Self{sources,current:Mutex::new(0)}}
    pub fn active_name(&self)->String{let index=self.current.lock().map(|v|*v).unwrap_or(usize::MAX);self.sources.get(index).map(|p|p.name()).unwrap_or("none").into()}
}
impl GpuProbe for Fallback{
    fn memory(&self)->Result<GpuMemory,ProbeError>{
        let mut current=self.current.lock().map_err(|_|ProbeError::Unavailable)?;
        while let Some(source)=self.sources.get(*current){
            match source.memory(){Ok(value)=>return Ok(value),Err(_)=>{tracing::warn!(event="gpu_fallback",probe=source.name(),"GPU-Messung fehlgeschlagen; nächste Quelle wird verwendet");*current+=1;}}
        }
        Err(ProbeError::Unavailable)
    }
    fn name(&self)->&str{"fallback"}
}
/// Die Reihenfolge entspricht der Konfiguration; fehlendes NVML wird als nicht verfügbar behandelt.
pub fn sources(vendor:GpuVendor,method:ProbeMethod,index:u32,budget:u64,root:PathBuf)->Fallback{
    let mut probes:Vec<Box<dyn GpuProbe>>=Vec::new();
    let selected=if method==ProbeMethod::Auto{match vendor{GpuVendor::Amd=>vec![ProbeMethod::Sysfs,ProbeMethod::RocmSmi],GpuVendor::Nvidia=>vec![ProbeMethod::Nvml,ProbeMethod::NvidiaSmi],_=>vec![]}}else{vec![method]};
    for method in selected{match method{
        ProbeMethod::Sysfs=>probes.push(Box::new(Sysfs{root:root.clone(),index})),
        ProbeMethod::NvidiaSmi=>probes.push(Box::new(Smi{program:"nvidia-smi".into(),index,kind:SmiKind::Nvidia})),
        ProbeMethod::RocmSmi=>probes.push(Box::new(Smi{program:"rocm-smi".into(),index,kind:SmiKind::Rocm})),_=>{}
    }}
    probes.push(Box::new(Accounting{budget,leases:AtomicU64::new(0)}));Fallback::new(probes)
}

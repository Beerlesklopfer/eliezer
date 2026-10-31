use eliezer_core::scheduler::*;
fn instance(key: &str, priority: i32, bytes: u64) -> Instance {
    Instance {
        key: key.into(),
        priority,
        bytes,
        state: State::Loaded,
        pinned: false,
        image: false,
        in_flight: 0,
        last_used_ms: 0,
    }
}
#[test]
fn eviction_is_atomic_and_obeys_priority_and_pin() {
    let request = instance("new", 5, 10);
    let mut old = instance("old", 1, 6);
    assert_eq!(plan_load(&[], &request, 10), Plan::Load);
    assert_eq!(
        plan_load(&[old.clone()], &request, 4),
        Plan::LoadAfterUnload(vec!["old".into()])
    );
    assert_eq!(plan_load(&[old.clone()], &request, 3), Plan::Reject);
    old.pinned = true;
    assert_eq!(plan_load(&[old.clone()], &request, 4), Plan::Reject);
    old.pinned = false;
    old.priority = 6;
    assert_eq!(plan_load(&[old.clone()], &request, 4), Plan::Reject);
    old.priority = 1;
    old.in_flight = 1;
    assert_eq!(plan_load(&[old], &request, 4), Plan::Wait);
}
#[test]
fn exclusivity_and_idle() {
    let mut request = instance("image", 0, 2);
    request.image = true;
    let mut old = instance("chat", 20, 4);
    assert_eq!(
        plan_load(&[old.clone()], &request, 20),
        Plan::LoadAfterUnload(vec!["chat".into()])
    );
    old.in_flight = 1;
    assert_eq!(plan_load(&[old.clone()], &request, 20), Plan::Wait);
    assert!(idle_candidates(&[old.clone()], 5000, 1000).is_empty());
    old.in_flight = 0;
    assert_eq!(idle_candidates(&[old.clone()], 5000, 1000), vec!["chat"]);
    old.pinned = true;
    assert_eq!(plan_load(&[old], &request, 20), Plan::Reject);
}
#[test]
fn all_state_edges() {
    let states = [
        State::Unloaded,
        State::Loading,
        State::Loaded,
        State::Unloading,
        State::Failed,
    ];
    let events = [
        Event::Load,
        Event::LoadDone,
        Event::LoadFailed,
        Event::Unload,
        Event::UnloadDone,
        Event::Crash,
        Event::Retry,
    ];
    let edges = [
        (State::Unloaded, Event::Load, State::Loading),
        (State::Loading, Event::LoadDone, State::Loaded),
        (State::Loading, Event::LoadFailed, State::Failed),
        (State::Loaded, Event::Crash, State::Failed),
        (State::Unloading, Event::Crash, State::Failed),
        (State::Loaded, Event::Unload, State::Unloading),
        (State::Unloading, Event::UnloadDone, State::Unloaded),
        (State::Failed, Event::Retry, State::Unloaded),
    ];
    for state in states {
        for event in events {
            assert_eq!(
                state.transition(event).ok(),
                edges
                    .iter()
                    .find(|(s, e, _)| *s == state && *e == event)
                    .map(|(_, _, s)| *s)
            );
        }
    }
}
#[test]
fn fifo_round_robin_batch_and_cancel() {
    let mut queue = Queue::new(4);
    for (id, key) in [(1, "a"), (2, "a"), (3, "b"), (4, "b")] {
        queue
            .push(
                key.into(),
                Queued {
                    id,
                    enqueued_ms: id,
                    value: (),
                },
            )
            .unwrap();
    }
    assert!(
        queue
            .push(
                "c".into(),
                Queued {
                    id: 5,
                    enqueued_ms: 0,
                    value: ()
                }
            )
            .is_err()
    );
    assert_eq!(queue.pop(None).unwrap().1.id, 1);
    assert_eq!(queue.pop(None).unwrap().1.id, 3);
    assert_eq!(queue.pop(Some("a")).unwrap().1.id, 2);
    assert_eq!(queue.remove_if(|i| i.enqueued_ms <= 4).len(), 1);
    assert!(queue.is_empty());
    queue
        .push(
            "a".into(),
            Queued {
                id: 6,
                enqueued_ms: 6,
                value: (),
            },
        )
        .unwrap();
    assert_eq!(queue.pop(None).unwrap().1.id, 6);
}
#[test]
fn placement_follows_the_image_priority() {
    use eliezer_core::config::DevicePolicy::*;
    // Diffusion always takes the GPU.
    assert_eq!(choose_place(Cpu, true, true, true), Place::Gpu);
    assert_eq!(choose_place(Auto, true, false, false), Place::Gpu);
    // Explicit devices are kept.
    assert_eq!(choose_place(Gpu, false, true, true), Place::Gpu);
    assert_eq!(choose_place(Cpu, false, false, false), Place::Cpu);
    // Auto: CPU only with a RAM budget and an image model in charge of the GPU.
    assert_eq!(choose_place(Auto, false, true, true), Place::Cpu);
    assert_eq!(choose_place(Auto, false, true, false), Place::Gpu);
    assert_eq!(choose_place(Auto, false, false, true), Place::Gpu);
}
#[test]
fn image_priority_holds_while_an_image_model_loads_loaded_or_waits() {
    let mut image = instance("img", 1, 10);
    image.image = true;
    assert!(image_priority(&[image.clone()], false));
    image.state = State::Loading;
    assert!(image_priority(&[image.clone()], false));
    image.state = State::Unloaded;
    assert!(!image_priority(&[image], false));
    assert!(image_priority(&[], true));
    assert!(!image_priority(&[instance("llm", 1, 1)], false));
}

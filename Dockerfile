FROM docker.io/library/rust:1.88-slim-bookworm AS build
WORKDIR /build
# The host toolchain file is not copied: the image toolchain applies here.
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY tests/m2 ./tests/m2
COPY packaging/config.example.toml ./packaging/config.example.toml
RUN cargo build --release --locked -p eliezer

FROM docker.io/library/debian:bookworm-slim
RUN mkdir -p /etc/eliezer /var/lib/eliezer/models /run/eliezer \
    && chown -R 10001:10001 /var/lib/eliezer /run/eliezer
COPY --from=build /build/target/release/eliezer /usr/local/bin/eliezer
COPY packaging/config.container.toml /etc/eliezer/config.toml
COPY LICENSE /usr/share/doc/eliezer/copyright
COPY --chmod=755 packaging/container-entrypoint.sh /usr/local/bin/eliezer-entrypoint
USER 10001:10001
EXPOSE 1234
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/local/bin/eliezer-entrypoint"]
CMD ["--config", "/etc/eliezer/config.toml"]

# Get started with a build env with Rust nightly
FROM rustlang/rust:nightly-bookworm as builder

# If you’re using stable, use this instead
# FROM rust:1.74-bullseye as builder

# Install CA certificates for SSL verification and build dependencies including WASM support
RUN apt-get update && apt-get install -y ca-certificates build-essential clang wasi-libc && rm -rf /var/lib/apt/lists/*

# Copy and trust the mkcert development CA for the proxy
COPY mkcert_development_CA_*.crt /usr/local/share/ca-certificates/
RUN update-ca-certificates

# Set environment variables for WASM compilation
ENV CFLAGS_wasm32_unknown_unknown="--sysroot=/usr/share/wasi-sysroot"

# Install cargo-leptos using cargo install
RUN cargo install cargo-leptos@0.2.17 --locked

# Add the WASM target
RUN rustup target add wasm32-unknown-unknown

# Make an /app dir, which everything will eventually live in
RUN mkdir -p /app
WORKDIR /app
# Ignore the ./target dir
COPY . .

# Build the app
RUN cargo leptos build --release -vv

FROM debian:bookworm-slim as runtime
WORKDIR /app
RUN apt-get update -y \
  && apt-get install -y --no-install-recommends  openssl ca-certificates \
  && apt-get autoremove -y \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

# Copy the server binary to the /app directory
COPY --from=builder /app/target/release/chatclm /app/

# /target/site contains our JS/WASM/CSS, etc.
COPY --from=builder /app/target/site /app/site

# Copy Cargo.toml if it’s needed at runtime
COPY --from=builder /app/Cargo.toml /app/

# Copy the zstd dictionary as it's needed at runtime
COPY --from=builder /app/model.zstd_dict /app/

# Set any required env variables and
ENV RUST_LOG="info"
ENV LEPTOS_SITE_ADDR="0.0.0.0:8080"
ENV LEPTOS_SITE_ROOT="site"
EXPOSE 8080

# -- NB: update binary name from "chatclm" to match your app name in Cargo.toml --
# Run the server
CMD ["/app/chatclm"]

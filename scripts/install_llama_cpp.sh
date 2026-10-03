#!/usr/bin/env bash
# Install a pinned official llama.cpp release binary into .tools/ (no Homebrew, no build).
# The sha256 is the GitHub release asset digest recorded at pin time. The script refuses to
# install on mismatch.
#
#   scripts/install_llama_cpp.sh                 # macOS arm64 (Metal)
#   LLAMA_ASSET=ubuntu-cuda-12.8-x64 scripts/install_llama_cpp.sh   # Linux CUDA bridge
set -euo pipefail

BUILD="${LLAMA_BUILD:-b11246}"
ASSET="${LLAMA_ASSET:-macos-arm64}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="${LLM_SHEET_TOOLS_DIR:-$ROOT/.tools}"
DEST="$TOOLS/llama.cpp/$BUILD-$ASSET"

case "$BUILD/$ASSET" in
  b11246/macos-arm64)          want=b463a0a8b0572e25b5b97647202b7a898a29120bcf3f19aa5a8db4b997f16708 ;;
  b11246/ubuntu-cuda-12.8-x64) want=7ec342733cc29e2fcea18ea445c1e96f93eaa7f06d9352f3d393dcf681f6ea5e ;;
  b11246/ubuntu-cuda-13.4-x64) want=101e06de269fa28c629b6047745603d1cedb8a9a80e844b4fb44a801d63a2971 ;;
  *) want="" ;;
esac
# CUDA runtime libraries shipped separately for the Linux CUDA builds (needed when the host
# image has no matching libcudart/libcublas on the loader path).
case "$BUILD/$ASSET" in
  b11246/ubuntu-cuda-12.8-x64) cudart_want=846d0789eedcc9bcc22c81ea8c45e3679e4bcce52051b3f953369e6708e41bd2 ;;
  *) cudart_want="" ;;
esac

if [[ -z "$want" ]]; then
  echo "No pinned sha256 for $BUILD/$ASSET. Add it from the GitHub release page first." >&2
  exit 2
fi

if [[ -x "$DEST/llama-server" ]]; then
  echo "already installed: $DEST/llama-server"
  exit 0
fi

name="llama-$BUILD-bin-$ASSET.tar.gz"
url="https://github.com/ggml-org/llama.cpp/releases/download/$BUILD/$name"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
echo "downloading $url"
curl -fL --retry 3 -o "$tmp/$name" "$url"
got="$(shasum -a 256 "$tmp/$name" | awk '{print $1}')"
if [[ "$got" != "$want" ]]; then
  echo "sha256 mismatch: got $got want $want" >&2
  exit 1
fi
mkdir -p "$DEST"
tar -xzf "$tmp/$name" -C "$tmp"
# Release tarballs contain a single top-level directory with binaries and dylibs.
src="$(find "$tmp" -name llama-server -type f -perm -u+x | head -1)"
cp -R "$(dirname "$src")"/. "$DEST"/
# Downloaded binaries carry the quarantine attribute on macOS. Remove it for local execution.
if [[ "$(uname -s)" == "Darwin" ]]; then
  xattr -dr com.apple.quarantine "$DEST" 2>/dev/null || true
fi
if [[ -n "$cudart_want" ]]; then
  cname="cudart-llama-$BUILD-bin-$ASSET.tar.gz"
  curl -fL --retry 3 -o "$tmp/$cname" "https://github.com/ggml-org/llama.cpp/releases/download/$BUILD/$cname"
  cgot="$(shasum -a 256 "$tmp/$cname" | awk '{print $1}')"
  [[ "$cgot" == "$cudart_want" ]] || { echo "cudart sha256 mismatch" >&2; exit 1; }
  mkdir -p "$tmp/cudart" && tar -xzf "$tmp/$cname" -C "$tmp/cudart"
  find "$tmp/cudart" -name "*.so*" -exec cp -P {} "$DEST"/ \;
  echo "(set LD_LIBRARY_PATH=$DEST if llama-server cannot find libcudart)"
fi
echo "$want  $name" > "$DEST/SOURCE_SHA256"
echo "installed: $DEST/llama-server"

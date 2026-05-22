#!/bin/sh

set -e

if [ -n "${DEBUG}" ]; then
  set -x
fi

# デフォルト設定
DEFAULT_INSTALL_PATH="/usr/local/bin"
REPO="tk-aria/kalidokit-rust"
BINARY_NAME="kalidokit-rust"

# ---------- ユーティリティ ----------

_latest_version() {
  curl -sSLf "https://api.github.com/repos/${REPO}/releases/latest" | \
    grep '"tag_name":' | \
    sed -E 's/.*"([^"]+)".*/\1/'
}

_detect_os() {
  os="$(uname -s)"
  case "$os" in
    Linux) echo "linux" ;;
    Darwin) echo "darwin" ;;
    CYGWIN*|MINGW*|MSYS*) echo "windows" ;;
    *) echo "Unsupported operating system: $os" 1>&2; return 1 ;;
  esac
  unset os
}

_detect_arch() {
  arch="$(uname -m)"
  case "$arch" in
    amd64|x86_64) echo "x86_64" ;;
    arm64|aarch64) echo "aarch64" ;;
    *) echo "Unsupported processor architecture: $arch" 1>&2; return 1 ;;
  esac
  unset arch
}

_get_target() {
  _os="$1"
  _arch="$2"
  case "$_os" in
    linux)   echo "${_arch}-unknown-linux-gnu" ;;
    darwin)  echo "aarch64-apple-darwin" ;;
    windows) echo "${_arch}-pc-windows-msvc" ;;
  esac
}

_get_ext() {
  _os="$1"
  case "$_os" in
    windows) echo "zip" ;;
    *) echo "tar.gz" ;;
  esac
}

_get_binary_file() {
  _os="$1"
  case "$_os" in
    windows) echo "${BINARY_NAME}.exe" ;;
    *) echo "${BINARY_NAME}" ;;
  esac
}

_download_url() {
  _version="$1"; _target="$2"; _ext="$3"
  echo "https://github.com/${REPO}/releases/download/${_version}/${BINARY_NAME}-${_version}-${_target}.${_ext}"
}

_resolve_install_path() {
  echo "${KALIDOKIT_INSTALL_PATH:-$DEFAULT_INSTALL_PATH}"
}

# ---------- download-models サブコマンド ----------

# モデルダウンロードURL
FACE_MODEL_URL="https://github.com/minh1311/mediapipe_onnx/raw/main/mediapipe/modules/face_landmark/face_landmark.onnx"
POSE_MODEL_URL="https://huggingface.co/opencv/pose_estimation_mediapipe/resolve/main/pose_estimation_mediapipe_2023mar.onnx"
HAND_MODEL_URL="https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/main/handpose_estimation_mediapipe_2023feb.onnx"
VRM_MODEL_URL="https://github.com/vrm-c/UniVRM/raw/master/Tests/Models/Alicia_vrm-0.51/AliciaSolid_vrm-0.51.vrm"
SMART_TURN_MODEL_URL="https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/onnx/model.onnx"

cmd_download_models() {
  models_dir="${KALIDOKIT_MODELS_PATH:-assets/models}"

  echo "Downloading model files to ${models_dir}..."

  if [ ! -d "$models_dir" ]; then
    echo "Creating models directory: $models_dir"
    mkdir -p "$models_dir"
  fi

  tmp_dir=$(mktemp -d)
  trap 'rm -rf "$tmp_dir"' EXIT

  # 1. Face landmark model
  if [ -f "$models_dir/face_landmark.onnx" ]; then
    echo "[skip] face_landmark.onnx already exists"
  else
    echo "[1/5] Downloading face landmark model..."
    if curl -sSLf -L "$FACE_MODEL_URL" -o "$models_dir/face_landmark.onnx"; then
      echo "  -> face_landmark.onnx ($(du -h "$models_dir/face_landmark.onnx" | cut -f1))"
    else
      echo "  [error] Failed to download face model" 1>&2
      rm -f "$models_dir/face_landmark.onnx"
    fi
  fi

  # 2. Pose landmark model
  if [ -f "$models_dir/pose_landmark.onnx" ]; then
    echo "[skip] pose_landmark.onnx already exists"
  else
    echo "[2/5] Downloading pose landmark model..."
    if curl -sSLf -L "$POSE_MODEL_URL" -o "$models_dir/pose_landmark.onnx"; then
      echo "  -> pose_landmark.onnx ($(du -h "$models_dir/pose_landmark.onnx" | cut -f1))"
    else
      echo "  [error] Failed to download pose model" 1>&2
      rm -f "$models_dir/pose_landmark.onnx"
    fi
  fi

  # 3. Hand landmark model
  if [ -f "$models_dir/hand_landmark.onnx" ]; then
    echo "[skip] hand_landmark.onnx already exists"
  else
    echo "[3/5] Downloading hand landmark model..."
    if curl -sSLf -L "$HAND_MODEL_URL" -o "$models_dir/hand_landmark.onnx"; then
      echo "  -> hand_landmark.onnx ($(du -h "$models_dir/hand_landmark.onnx" | cut -f1))"
    else
      echo "  [error] Failed to download hand model" 1>&2
      rm -f "$models_dir/hand_landmark.onnx"
    fi
  fi

  # 4. Default VRM avatar
  if [ -f "$models_dir/default_avatar.vrm" ]; then
    echo "[skip] default_avatar.vrm already exists"
  else
    echo "[4/5] Downloading default VRM avatar..."
    if curl -sSLf -L "$VRM_MODEL_URL" -o "$models_dir/default_avatar.vrm"; then
      echo "  -> default_avatar.vrm ($(du -h "$models_dir/default_avatar.vrm" | cut -f1))"
    else
      echo "  [error] Failed to download VRM avatar" 1>&2
      rm -f "$models_dir/default_avatar.vrm"
    fi
  fi

  # 5. Smart turn model
  if [ -f "$models_dir/smart_turn_v3.onnx" ]; then
    echo "[skip] smart_turn_v3.onnx already exists"
  else
    echo "[5/5] Downloading smart turn model..."
    if curl -sSLf -L "$SMART_TURN_MODEL_URL" -o "$models_dir/smart_turn_v3.onnx"; then
      echo "  -> smart_turn_v3.onnx ($(du -h "$models_dir/smart_turn_v3.onnx" | cut -f1))"
    else
      echo "  [error] Failed to download smart turn model" 1>&2
      rm -f "$models_dir/smart_turn_v3.onnx"
    fi
  fi

  echo ""
  echo "Model download complete!"
  echo "Files in ${models_dir}:"
  ls -lh "$models_dir/" 2>/dev/null || true
}

# ---------- install サブコマンド ----------

cmd_install() {
  # バージョン決定
  if [ -z "${KALIDOKIT_VERSION}" ]; then
    echo "Getting latest version..."
    KALIDOKIT_VERSION=$(_latest_version)
    if [ -z "${KALIDOKIT_VERSION}" ]; then
      echo "Failed to get latest version" 1>&2
      return 1
    fi
  fi

  install_path="$(_resolve_install_path)"
  detected_os="$(_detect_os)"
  detected_arch="$(_detect_arch)"
  target="$(_get_target "$detected_os" "$detected_arch")"
  ext="$(_get_ext "$detected_os")"
  binary="$(_get_binary_file "$detected_os")"
  download_url="$(_download_url "$KALIDOKIT_VERSION" "$target" "$ext")"

  echo "Installing ${BINARY_NAME} ${KALIDOKIT_VERSION} for ${detected_os}/${detected_arch} (${target})..."
  echo "Download URL: $download_url"

  # インストールディレクトリ作成
  if [ ! -d "$install_path" ]; then
    echo "Creating install directory: $install_path"
    mkdir -p "$install_path"
  fi

  # 一時ディレクトリ
  tmp_dir=$(mktemp -d)
  trap 'rm -rf "$tmp_dir"' EXIT

  # ダウンロード
  echo "Downloading..."
  if ! curl -sSLf "$download_url" -o "$tmp_dir/archive.${ext}"; then
    echo "Failed to download from: $download_url" 1>&2
    echo "Check if version ${KALIDOKIT_VERSION} exists for ${target}" 1>&2
    return 1
  fi

  # 展開
  echo "Extracting..."
  case "$ext" in
    tar.gz) tar -xzf "$tmp_dir/archive.tar.gz" -C "$tmp_dir" ;;
    zip)    unzip -q "$tmp_dir/archive.zip" -d "$tmp_dir" ;;
  esac

  # バイナリ検索
  archive_dir="${BINARY_NAME}-${KALIDOKIT_VERSION}-${target}"
  if [ -f "$tmp_dir/${archive_dir}/${binary}" ]; then
    binary_path="$tmp_dir/${archive_dir}/${binary}"
  elif [ -f "$tmp_dir/${binary}" ]; then
    binary_path="$tmp_dir/${binary}"
  else
    echo "Binary not found in archive. Expected: ${archive_dir}/${binary}" 1>&2
    return 1
  fi

  # 配置 (権限がない場合は ~/.local/bin にフォールバック)
  _installed=0
  if cp "$binary_path" "$install_path/$binary" 2>/dev/null; then
    chmod 755 "$install_path/$binary"
    _installed=1
  else
    # フォールバック: ~/.local/bin
    fallback_path="$HOME/.local/bin"
    if [ "$install_path" != "$fallback_path" ]; then
      echo "Permission denied for $install_path — falling back to $fallback_path"
      mkdir -p "$fallback_path"
      if cp "$binary_path" "$fallback_path/$binary" 2>/dev/null; then
        chmod 755 "$fallback_path/$binary"
        install_path="$fallback_path"
        _installed=1
      fi
    fi
  fi

  if [ "$_installed" -eq 0 ]; then
    echo "Failed to copy binary to $install_path" 1>&2
    echo "Try: sudo sh setup.sh install" 1>&2
    return 1
  fi

  # assets
  if [ -d "$tmp_dir/${archive_dir}/assets" ]; then
    assets_dest="$(dirname "$install_path")/../share/${BINARY_NAME}"
    if mkdir -p "$assets_dest" 2>/dev/null; then
      cp -r "$tmp_dir/${archive_dir}/assets" "$assets_dest/" 2>/dev/null && \
        echo "Assets installed to: $assets_dest/assets"
    fi
  fi

  echo ""
  echo "${BINARY_NAME} ${KALIDOKIT_VERSION} installed successfully!"
  echo "  Binary: $install_path/$binary"

  # PATH に含まれていない場合のヒント
  case ":$PATH:" in
    *":$install_path:"*) ;;
    *)
      echo ""
      echo "NOTE: $install_path is not in your PATH."
      echo "Add it with:"
      echo "  export PATH=\"$install_path:\$PATH\""
      echo ""
      echo "To make it permanent, add the line above to your ~/.zshrc or ~/.bashrc"
      ;;
  esac

  echo ""
  echo "Run '${BINARY_NAME} --help' to get started."
}

# ---------- uninstall サブコマンド ----------

cmd_uninstall() {
  install_path="$(_resolve_install_path)"
  detected_os="$(_detect_os)"
  binary="$(_get_binary_file "$detected_os")"
  binary_full="$install_path/$binary"

  if [ ! -f "$binary_full" ]; then
    echo "${BINARY_NAME} is not installed at $binary_full" 1>&2
    echo "If installed elsewhere, set KALIDOKIT_INSTALL_PATH" 1>&2
    return 1
  fi

  echo "Removing ${BINARY_NAME} from $binary_full ..."
  if ! rm -f "$binary_full"; then
    echo "Failed to remove $binary_full. Try: sudo sh setup.sh uninstall" 1>&2
    return 1
  fi

  # assets 削除
  assets_dir="$(dirname "$install_path")/../share/${BINARY_NAME}"
  if [ -d "$assets_dir" ]; then
    echo "Removing assets from $assets_dir ..."
    rm -rf "$assets_dir"
  fi

  echo ""
  echo "${BINARY_NAME} has been uninstalled."
}

# ---------- usage ----------

usage() {
  cat <<EOF
Usage: setup.sh <command> [options]

Commands:
  install           Download and install ${BINARY_NAME}
  uninstall         Remove ${BINARY_NAME}
  download-models   Download required ML models and VRM avatar

Environment variables:
  KALIDOKIT_VERSION       Version to install (default: latest)
  KALIDOKIT_INSTALL_PATH  Install directory (default: /usr/local/bin)
  KALIDOKIT_MODELS_PATH   Models directory (default: assets/models)
  DEBUG                   Enable verbose output

Examples:
  # Install latest
  curl -sSLf https://raw.githubusercontent.com/tk-aria/kalidokit-rust/main/scripts/setup.sh | sh -s install

  # Install specific version
  curl -sSLf https://raw.githubusercontent.com/tk-aria/kalidokit-rust/main/scripts/setup.sh | KALIDOKIT_VERSION=v0.1.0 sh -s install

  # Install to custom path
  curl -sSLf https://raw.githubusercontent.com/tk-aria/kalidokit-rust/main/scripts/setup.sh | KALIDOKIT_INSTALL_PATH=~/.local/bin sh -s install

  # Download ML models
  sh setup.sh download-models

  # Download models to custom path
  KALIDOKIT_MODELS_PATH=./my-models sh setup.sh download-models

  # Uninstall
  curl -sSLf https://raw.githubusercontent.com/tk-aria/kalidokit-rust/main/scripts/setup.sh | sh -s uninstall
EOF
}

# ---------- エントリポイント ----------

main() {
  command="${1:-}"

  case "$command" in
    install)          cmd_install ;;
    uninstall)        cmd_uninstall ;;
    download-models)  cmd_download_models ;;
    -h|--help|help)   usage ;;
    "")
      # サブコマンドなし → デフォルトで install (後方互換)
      cmd_install
      ;;
    *)
      echo "Unknown command: $command" 1>&2
      echo "" 1>&2
      usage 1>&2
      return 1
      ;;
  esac
}

main "$@"

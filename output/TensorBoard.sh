SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
SUMMARY_DIR=$(realpath -m "${SCRIPT_DIR}/summary")
PORT=6067
echo $SUMMARY_DIR
echo http://localhost:${PORT}/
tensorboard --logdir=${SUMMARY_DIR} --port=${PORT} --load_fast=false --bind_all
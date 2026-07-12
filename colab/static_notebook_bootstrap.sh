#!/usr/bin/env bash
set -euo pipefail

bash colab/bootstrap.sh --colmap-cuda

python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "import gsplat; print('gsplat', getattr(gsplat, '__version__', 'unknown'))"
colmap -h >/dev/null
ffmpeg -version >/dev/null
python -c "import pydantic; print('pydantic', pydantic.__version__)"
python -c "import nbformat; print('nbformat', nbformat.__version__)"

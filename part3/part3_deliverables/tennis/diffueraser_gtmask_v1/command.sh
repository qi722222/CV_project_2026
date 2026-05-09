conda run -n diffueraser_env python3 part3/inpainting/prepare_diffueraser_inputs.py --seq tennis --version v1
conda run -n diffueraser_env python3 part3/inpainting/run_diffueraser_gtmask.py --seq tennis --version v1

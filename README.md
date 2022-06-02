
**Notes**
- The jupyter notebook OPT175B can be used to run inference on the 175B model (in half precision)
- This uses model parallelism and splits the model into 9 A100 GPUs
- The notebook should be runnable on the server after installing the dependencies below
- The notebook reads model weights from `/home/llajan/OPT-175B-HF`

**Dependencies**

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install transformers accelerate
pip3 install ipywidgets jupyterlab

**Other details**

This part can be skipped as the format conversion has already been done and is only included as a reference.

Steps to convert OPT weights from Meta's format to HuggingFace's format

1) Obtain OPT 175B weights from Meta
2) Convert model parts into a single file
```
python metaseq/scripts/consolidate_fsdp_shards.py ${FOLDER_PATH}/checkpoint_last --new-arch-name transformer_lm_gpt --save-prefix ${FOLDER_PATH}/consolidated
```

3) Use conversion script from HF [2] to convert this into HF format
```
python convert_opt_original_pytorch_checkpoint_to_pytorch.py --pytorch_dump_folder_path <path/to/dump/hf/model> --hf_config config.json --fairseq_path <path/to/restored.py>
```

**References**

[1]  The jupyter notebook is an adaptation of this colab from HF (https://colab.research.google.com/drive/14wnxMvD9zsiBQo2FtTpxn6w2cpXCcb-7#scrollTo=MoGXqv_lmVhN&uniqifier=1)

[2] Format conversion script https://github.com/facebookresearch/metaseq/issues/98#issuecomment-1125859328)

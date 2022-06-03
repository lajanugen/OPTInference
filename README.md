This repository details the steps to run inference on Meta's OPT 175B model using the HuggingFace library.

## Notes
- The jupyter notebook `OPT175B.ipynb` can be used to run inference on the OPT 175B model
- This uses model parallelism (splits model into 9 A100 GPUs on internal server)
- The notebook should be runnable on the server after installing the dependencies below
- `weights_path` should point to the path where model weights are stored (in HF format)

## Dependencies
```
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install transformers accelerate
pip3 install ipywidgets jupyterlab
```

## Other details (format conversion)

Steps to convert OPT weights from Meta's format to HuggingFace's format (Can be skipped if converted weights are available)

1) Obtain OPT 175B weights from Meta
2) Convert model parts into a single file using [metaseq](https://github.com/facebookresearch/metaseq)
```
python metaseq/scripts/consolidate_fsdp_shards.py ${FOLDER_PATH}/checkpoint_last --new-arch-name transformer_lm_gpt --save-prefix ${FOLDER_PATH}/consolidated
```

3) Use [conversion script](https://github.com/facebookresearch/metaseq/issues/98#issuecomment-11258593280) from HF to convert this into HF format
```
python convert_opt_original_pytorch_checkpoint_to_pytorch.py --pytorch_dump_folder_path <path/to/dump/hf/model> --hf_config config.json --fairseq_path <path/to/restored.py>
```

## References

[1]  The jupyter notebook is an adaptation of this [colab](https://colab.research.google.com/drive/14wnxMvD9zsiBQo2FtTpxn6w2cpXCcb-7#scrollTo=MoGXqv_lmVhN&uniqifier=1) from HF

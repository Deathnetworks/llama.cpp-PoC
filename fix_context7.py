with open("llama.cpp-PoC/src/llama-model.cpp", "r") as f:
    ccontent = f.read()

# fix llama_model_default_params
old_params = """llama_model_params llama_model_default_params() {
    llama_model_params result = {
        /*.ffn_mode                    =*/ 0,
        /*.ffn_file                    =*/ nullptr,
        /*.devices                     =*/ nullptr,
        /*.tensor_buft_overrides       =*/ nullptr,
        /*.n_gpu_layers                =*/ -1,
        /*.split_mode                  =*/ LLAMA_SPLIT_MODE_LAYER,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_direct_io               =*/ false,
        /*.use_mlock                   =*/ false,
        /*.check_tensors               =*/ false,
        /*.use_extra_bufts             =*/ true,
        /*.no_host                     =*/ false,
        /*.no_alloc                    =*/ false,
    };"""

new_params = """llama_model_params llama_model_default_params() {
    llama_model_params result = {
        /*.ffn_mode                    =*/ 0,
        /*.ffn_file                    =*/ nullptr,
        /*.devices                     =*/ nullptr,
        /*.tensor_buft_overrides       =*/ nullptr,
        /*.n_gpu_layers                =*/ -1,
        /*.split_mode                  =*/ LLAMA_SPLIT_MODE_LAYER,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_direct_io               =*/ false,
        /*.use_mlock                   =*/ false,
        /*.check_tensors               =*/ false,
        /*.use_extra_bufts             =*/ true,
        /*.no_host                     =*/ false,
        /*.no_alloc                    =*/ false,
    };"""

ccontent = ccontent.replace(old_params, new_params)

with open("llama.cpp-PoC/src/llama-model.cpp", "w") as f:
    f.write(ccontent)

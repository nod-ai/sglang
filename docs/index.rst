SGLang Documentation
====================

SGLang is a fast serving framework for large language models and vision language models.
It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language.
The core features include:

- **Fast Backend Runtime**: Provides efficient serving with RadixAttention for prefix caching, zero-overhead CPU scheduler, continuous batching, token attention (paged attention), speculative decoding, tensor parallelism, chunked prefill, structured outputs, quantization (FP8/INT4/AWQ/GPTQ), and multi-lora batching.
- **Flexible Frontend Language**: Offers an intuitive interface for programming LLM applications, including chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
- **Extensive Model Support**: Supports a wide range of generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.), embedding models (e5-mistral, gte, mcdse) and reward models (Skywork), with easy extensibility for integrating new models.
- **Active Community**: SGLang is open-source and backed by an active community with industry adoption.

.. toctree::
   :maxdepth: 1
   :caption: Installation

   start/install.md

.. toctree::
   :maxdepth: 1
   :caption: Backend Tutorial

   references/deepseek
   references/llama4
   backend/send_request.ipynb
   backend/openai_api_completions.ipynb
   backend/openai_api_vision.ipynb
   backend/openai_api_embeddings.ipynb
   backend/native_api.ipynb
   backend/offline_engine_api.ipynb
   backend/structured_outputs.ipynb
   backend/speculative_decoding.ipynb
   backend/function_calling.ipynb
   backend/server_arguments.md

.. toctree::
   :maxdepth: 1
   :caption: Advanced Backend Configurations

   backend/server_arguments.md
   backend/sampling_params.md
   backend/hyperparameter_tuning.md
   backend/attention_backend.md

.. toctree::
   :maxdepth: 1
   :caption: Supported Models

   supported_models/generative_models.md
   supported_models/multimodal_language_models.md
   supported_models/embedding_models.md
   supported_models/reward_models.md
   supported_models/support_new_models.md

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   backend/speculative_decoding.ipynb
   backend/structured_outputs.ipynb
   backend/function_calling.ipynb
   backend/separate_reasoning.ipynb
   backend/structured_outputs_for_reasoning_models.ipynb
   backend/custom_chat_template.md
   backend/quantization.md
   backend/lora.ipynb
   backend/pd_disaggregation.md

.. toctree::
   :maxdepth: 1
   :caption: Frontend Tutorial

   frontend/frontend.ipynb
   frontend/choices_methods.md

.. toctree::
   :maxdepth: 1
   :caption: SGLang Router

   router/router.md

.. toctree::
      :maxdepth: 1
      :caption: References

   references/supported_models.md
   references/sampling_params.md
   references/hyperparameter_tuning.md
   references/benchmark_and_profiling.md
   references/custom_chat_template.md
   references/deepseek.md
   references/llama_405B.md
   references/modelscope.md
   references/contribution_guide.md
   references/troubleshooting.md
   references/faq.md
   references/learn_more.md

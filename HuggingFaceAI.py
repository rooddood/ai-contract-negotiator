import time

class HuggingFaceAI:
    def __init__(self):
        self.is_torch = False
        self.is_mlx = False
        self.is_tensorflow = False
        self.is_jax = False
        self.is_huggingface = False

    def check_installed_software(self):
        try:
            import torch
            print(f"Pytorch version: {torch.__version__}")
            self.is_torch = True
            if torch.backends.mps.is_available():
                print("Apple Metal MPS acceleration ok.")
            else:
                print("Your version of Pytorch does not support MPS, Pytorch will be slow.")
        except ImportError:
            print("Pytorch is not installed. Please install pytorch!")

        try:
            import mlx.core as mx
            print(f"MLX version: {mx.__version__}")
            self.is_mlx = True
            print("Apple MLX framework is installed ok")
        except ImportError:
            print("MLX is not installed, it's optional, so this is not a fatal error.")

        try:
            import tensorflow as tf
            print(f"Tensorflow version: {tf.__version__}")
            self.is_tensorflow = True
            devs = tf.config.list_physical_devices('GPU')
            if not devs:
                print("You have not installed the metal drivers, tensorflow will be slow")
            else:
                print(f"GPU support ok: {devs}")
        except ImportError:
            print("Tensorflow not installed, but it's optional, so this is not a fatal error.")

        try:
            import jax
            self.is_jax = True
            device_type = jax.devices()[0].device_kind
            print(f"JAX is installed and is using: {device_type}, ok")
        except ImportError:
            print("JAX is not installed, it's optional, so this is not a fatal error.")

        try:
            import transformers
            print(f"Transformers version: {transformers.__version__}")
            self.is_huggingface = True
        except ImportError as e:
            print(f"HuggingFace transformers is not installed. This won't work! {e}")

    def setup_pipeline(self, task="text-generation", model="gpt2"):
        """
        Sets up the HuggingFace pipeline for a specified task and model.
        """
        if not self.is_huggingface or not self.is_torch:
            print("HuggingFace and PyTorch are required but not properly installed.")
            return None

        from transformers import pipeline
        try:
            print(f"Initializing HuggingFace pipeline for task: {task}, model: {model}")
            return pipeline(task, model=model, framework="pt")
        except Exception as e:
            print(f"Error initializing HuggingFace pipeline: {e}")
            return None

    def run_sentiment_analysis(self):
        if not (self.is_huggingface and self.is_torch):
            print("The minimal software is not installed. Please check that PyTorch and HuggingFace are installed.")
            return

        from transformers import pipeline
        print("All looks good, let's try a simple sentiment analysis:")
        nlp = pipeline("sentiment-analysis", framework='pt')
        result = nlp("We are very happy to show you the 🤗 Transformers library.")
        print(result)

    def run_benchmarks(self):
        import numpy as np
        import jax
        from jax import jit
        import jax.numpy as jnp
        import torch
        import mlx.core as mx

        # Numpy Benchmark
        x = np.random.rand(2048, 2048).astype(dtype=np.float32) / 5.0
        def bench_func(x):
            for i in range(3):
                x = (np.matmul(x, x) + x) / 1000.0
            return x
        print("Numpy Benchmark:")
        start = time.time()
        bench_func(x)
        end = time.time()
        print(f"Execution time: {end - start:.6f} seconds")

        # JAX Benchmark
        xj = jnp.array(x)
        def bench_func_j(x):
            for i in range(3):
                x = (jnp.matmul(x, x) + x) / 1000.0
            return x
        print("JAX Benchmark:")
        start = time.time()
        jit(bench_func_j)(xj).block_until_ready()
        end = time.time()
        print(f"Execution time: {end - start:.6f} seconds")

        # Torch Benchmark
        xt = torch.tensor(x)
        def bench_func_t(x):
            for i in range(3):
                x = (torch.matmul(x, x) + x) / 1000.0
            return x
        print("Torch Benchmark:")
        start = time.time()
        bench_func_t(xt)
        end = time.time()
        print(f"Execution time: {end - start:.6f} seconds")

        bench_func_tc = torch.compile(bench_func_t)
        print("Torch Compile Benchmark:")
        start = time.time()
        bench_func_tc(xt)
        end = time.time()
        print(f"Execution time: {end - start:.6f} seconds")

        # MLX Benchmark
        xm = mx.array(x)
        def bench_func_m(x1):
            for _ in range(3):
                x1 = (mx.matmul(x1, x1) + x1) / mx.array(1000.0)
            return x1
        print("MLX Benchmark:")
        start = time.time()
        mx.eval(bench_func_m(xm))
        end = time.time()
        print(f"Execution time: {end - start:.6f} seconds")

        bench_func_mc = mx.compile(bench_func_m)
        print("MLX Compile Benchmark:")
        start = time.time()
        mx.eval(bench_func_mc(xm))
        end = time.time()
        print(f"Execution time: {end - start:.6f} seconds")

    def run_chatbot(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging

        logging.set_verbosity_error()

        model_names = ["microsoft/DialoGPT-small", "microsoft/DialoGPT-medium", "microsoft/DialoGPT-large"]
        use_model_index = 2  # Change 0: small model, 1: medium, 2: large model
        model_name = model_names[use_model_index]

        tokenizer = AutoTokenizer.from_pretrained(model_name, framework='pt')
        model = AutoModelForCausalLM.from_pretrained(model_name)

        def reply(input_text, history=None):
            new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
            bot_input_ids = torch.cat([history, new_user_input_ids], dim=-1) if history is not None else new_user_input_ids
            chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True), chat_history_ids

        history = None
        first = True
        while True:
            if first:
                first = False
                print("Please press enter (not SHIFT-enter) after your input:")
            input_text = input("> ")
            if input_text in ["", "bye", "quit", "exit"]:
                break
            reply_text, history_new = reply(input_text, history)
            history = history_new
            if history.shape[1] > 80:
                old_shape = history.shape
                history = history[:, -80:]
                print(f"History cut from {old_shape} to {history.shape}")
            print(f"D_GPT: {reply_text}")

    def run_text2text_generation(self):
        if not (self.is_huggingface and self.is_torch):
            print("The minimal software is not installed. Please check that PyTorch and HuggingFace are installed.")
            return

        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

        print("Initializing text2text-generation pipeline with google/flan-t5-large model...")
        try:
            # Using pipeline as a high-level helper
            pipe = pipeline("text2text-generation", model="google/flan-t5-large")
            print("Pipeline initialized successfully.")

            # Example usage of the pipeline
            input_text = "Translate English to French: How are you?"
            result = pipe(input_text)
            print(f"Pipeline output: {result}")

            # Loading model and tokenizer directly
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
            print("Model and tokenizer loaded successfully.")

            # Example usage of the model and tokenizer
            inputs = tokenizer(input_text, return_tensors="pt")
            outputs = model.generate(**inputs)
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Direct model output: {decoded_output}")
        except Exception as e:
            print(f"Error initializing or using google/flan-t5-large model: {e}")

    def run_deepseek_v3(self):
        if not (self.is_huggingface and self.is_torch):
            print("The minimal software is not installed. Please check that PyTorch and HuggingFace are installed.")
            return

        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

        print("Initializing text-generation pipeline with deepseek-ai/DeepSeek-V3-0324 model...")
        try:
            # Using pipeline as a high-level helper
            messages = [
                {"role": "user", "content": "Who are you?"},
            ]
            pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-V3-0324", trust_remote_code=True, device=-1)  # Force CPU mode
            result = pipe(messages)
            print(f"Pipeline output: {result}")

            # Loading model and tokenizer directly
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-0324", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V3-0324", trust_remote_code=True)

            # Fix rope_scaling configuration if needed
            if hasattr(model.config, 'rope_scaling'):
                rope_scaling = model.config.rope_scaling
                if not isinstance(rope_scaling.get('factor', 1), float) or rope_scaling['factor'] < 1:
                    rope_scaling['factor'] = 1.0  # Set a valid default
                if not isinstance(rope_scaling.get('beta_fast', 1.0), float):
                    rope_scaling['beta_fast'] = 1.0
                if not isinstance(rope_scaling.get('beta_slow', 1.0), float):
                    rope_scaling['beta_slow'] = 1.0
                print("Fixed rope_scaling configuration.")

            print("Model and tokenizer loaded successfully.")

            # Example usage of the model and tokenizer
            input_text = "Who are you?"
            inputs = tokenizer(input_text, return_tensors="pt")
            outputs = model.generate(**inputs)
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Direct model output: {decoded_output}")
        except Exception as e:
            print(f"Error initializing or using deepseek-ai/DeepSeek-V3-0324 model: {e}")

    def main(self):
        self.check_installed_software()
        self.run_sentiment_analysis()
        self.run_benchmarks()
        self.run_chatbot()
        self.run_text2text_generation()
        self.run_deepseek_v3()


if __name__ == "__main__":
    ai = HuggingFaceAI()
    ai.main()
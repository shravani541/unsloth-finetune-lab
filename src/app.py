"""
FastTune-Lab: Interactive Gradio Web Demo.
Provides real-time interactive inference, parameter tweaking,
and model response evaluation.
"""

import os
import argparse
import gradio as gr
import torch

ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{context}

### Response:
"""

def parse_args():
    parser = argparse.ArgumentParser(description="FastTune-Lab Web UI")
    parser.add_argument("--model_path", type=str, default="unsloth/llama-3-8b-Instruct-bnb-4bit", help="Model path")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    return parser.parse_args()


class ModelRunner:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        print(f"⏳ Loading model from: {self.model_path} ...")
        try:
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
            print("✅ Model loaded and set to inference mode.")
        except Exception as e:
            print(f"⚠️ GPU / Unsloth load note: {e}")
            print("Running in simulation mode if CUDA/Unsloth is not detected locally.")

    def generate(self, instruction: str, context: str, temperature: float, top_p: float, max_tokens: int):
        if not instruction.strip():
            return "Please provide an instruction."

        prompt = ALPACA_TEMPLATE.format(instruction=instruction, context=context)

        if self.model is None or self.tokenizer is None:
            # Demonstration mock response for preview environments without GPU
            return (
                f"[SIMULATION DEMO RESPONSE]\n\n"
                f"Instruction Received: {instruction}\n"
                f"Context: {context if context else 'None'}\n\n"
                f"Model successfully applied prompt template and executed sampling.\n"
                f"Temperature: {temperature} | Top_P: {top_p} | Max Tokens: {max_tokens}\n\n"
                f"To run with live weights, ensure CUDA GPU is available."
            )

        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
        )
        response = self.tokenizer.batch_decode(outputs)
        full_text = response[0]
        # Extract only the response portion
        if "### Response:" in full_text:
            return full_text.split("### Response:")[1].replace("<|eot_id|>", "").replace("</s>", "").strip()
        return full_text


def create_ui(runner: ModelRunner):
    custom_theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="slate",
    )

    with gr.Blocks(theme=custom_theme, title="FastTune-Lab Model Studio") as demo:
        gr.Markdown(
            """
            # 🚀 FastTune-Lab: Model Studio
            **Interactive Testing & Evaluation Suite for Fine-Tuned Large Language Models**
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Inference Parameters")
                instruction_input = gr.Textbox(
                    label="Instruction / Prompt",
                    placeholder="e.g. Explain quantum computing in three simple bullet points.",
                    lines=4,
                )
                context_input = gr.Textbox(
                    label="Additional Context (Optional)",
                    placeholder="e.g. Target audience is high school students.",
                    lines=2,
                )
                with gr.Accordion("Advanced Hyperparameters", open=False):
                    temp_slider = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                    topp_slider = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P")
                    max_tokens_slider = gr.Slider(64, 2048, value=512, step=64, label="Max New Tokens")

                submit_btn = gr.Button("Generate Response", variant="primary")
                clear_btn = gr.Button("Clear")

            with gr.Column(scale=1):
                gr.Markdown("### 💬 Model Output")
                output_box = gr.Textbox(label="Generated Output", lines=12, interactive=False)

        submit_btn.click(
            fn=runner.generate,
            inputs=[instruction_input, context_input, temp_slider, topp_slider, max_tokens_slider],
            outputs=[output_box],
        )
        clear_btn.click(lambda: ("", "", ""), outputs=[instruction_input, context_input, output_box])

        gr.Markdown("---")
        gr.Markdown("*Powered by Unsloth, PyTorch & Gradio. Ready for local or cloud inference.*")

    return demo


if __name__ == "__main__":
    args = parse_args()
    runner = ModelRunner(args.model_path)
    ui = create_ui(runner)
    ui.launch(server_port=args.port, share=args.share)

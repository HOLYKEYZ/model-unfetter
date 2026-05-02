"""
3 Levels of Quantization module.
Automates conversion of the model using llama.cpp to three targeted hardware profiles.
"""

import os
import subprocess
import logging

logger = logging.getLogger(__name__)

class Quantizer:
    def __init__(self, llama_cpp_dir: str):
        self.llama_cpp_dir = llama_cpp_dir
        self.quantize_bin = os.path.join(llama_cpp_dir, "llama-quantize")
        if os.name == 'nt':
            self.quantize_bin += ".exe"

        if not os.path.exists(self.quantize_bin):
            logger.warning(f"Could not find llama-quantize at {self.quantize_bin}. Quantization will fail.")

    def run_quantization(self, input_gguf: str, output_gguf: str, qtype: str):
        logger.info(f"Quantizing to {qtype}: {output_gguf}")
        
        cmd = [
            self.quantize_bin,
            input_gguf,
            output_gguf,
            qtype
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Quantization to {qtype} successful.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Quantization failed: {e.stderr}")
            raise

    def build_3_levels(self, input_f16_gguf: str, output_dir: str, model_name: str):
        """
        Builds the three standardized tiers for low hardware:
        1. Q4_K_M (8GB RAM, High Compression)
        2. Q8_0 (16GB RAM, Balanced)
        3. F16 (Studio Grade, No Compression)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        levels = [
            ("Q4_K_M", "q4_k_m"),
            ("Q8_0", "q8_0")
        ]
        
        # Note: Input is already F16, so we just copy it for the 3rd level
        f16_dest = os.path.join(output_dir, f"{model_name}-f16.gguf")
        if not os.path.exists(f16_dest) and input_f16_gguf != f16_dest:
            logger.info("Copying F16 base model for Level 3...")
            import shutil
            shutil.copy2(input_f16_gguf, f16_dest)

        for label, qtype in levels:
            out_path = os.path.join(output_dir, f"{model_name}-{qtype}.gguf")
            if not os.path.exists(out_path):
                self.run_quantization(input_f16_gguf, out_path, qtype)
            else:
                logger.info(f"Skipping {qtype}, file already exists.")

        logger.info("3 Levels of Quantization Complete.")
        logger.info(f"  Level 1 (8GB): {model_name}-q4_k_m.gguf")
        logger.info(f"  Level 2 (16GB): {model_name}-q8_0.gguf")
        logger.info(f"  Level 3 (Studio): {model_name}-f16.gguf")

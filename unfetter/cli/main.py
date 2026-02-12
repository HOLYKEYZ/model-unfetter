"""
CLI entry point for Model Unfetter.

Usage:
    unfetter ablate <model> [options]
    unfetter resume <checkpoint>
    unfetter compare <original> <ablated>
    unfetter validate <model>
    unfetter info
"""

import sys
import logging

import click

from unfetter import __version__


def setup_logging(verbose: bool = False):
    """Configure logging level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )


BANNER = r"""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ██╗   ██╗███╗   ██╗███████╗███████╗████████╗████████╗  ║
║   ██║   ██║████╗  ██║██╔════╝██╔════╝╚══██╔══╝╚══██╔══╝  ║
║   ██║   ██║██╔██╗ ██║█████╗  █████╗     ██║      ██║     ║
║   ██║   ██║██║╚██╗██║██╔══╝  ██╔══╝     ██║      ██║     ║
║   ╚██████╔╝██║ ╚████║██║     ███████╗   ██║      ██║     ║
║    ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚══════╝   ╚═╝      ╚═╝     ║
║                                                           ║
║   Model Unfetter — Directional Ablation Framework         ║
║   ⚠️  For AI Safety Research & Red Teaming Only            ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""


@click.group()
@click.version_option(version=__version__, prog_name="unfetter")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx, verbose):
    """Model Unfetter — Multi-tier model unalignment framework."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@cli.command()
@click.argument("model_path")
@click.option("--backend", type=click.Choice(["auto", "cpu", "gpu", "distributed"]),
              default="auto", help="Processing backend")
@click.option("--output", "-o", default=None, help="Output directory for ablated model")
@click.option("--strength", "-s", type=float, default=1.0,
              help="Ablation strength (0.0-1.0)")
@click.option("--layers", "-l", default="auto",
              help="Layer specification (auto, all, -8:-1, 20,25,30)")
@click.option("--targets", default="refusal",
              help="Comma-separated ablation targets")
@click.option("--ram", type=int, default=None, help="RAM limit in GB")
@click.option("--vram", type=float, default=None, help="VRAM limit in GB")
@click.option("--checkpoint-every", type=int, default=5,
              help="Save checkpoint every N layers")
@click.option("--checkpoint-dir", default=None,
              help="Checkpoint directory")
@click.option("--validate/--no-validate", default=False,
              help="Run validation after ablation")
@click.option("--cache-vectors/--no-cache-vectors", default=True,
              help="Cache computed refusal vectors")
@click.option("--format", "output_format",
              type=click.Choice(["safetensors", "pytorch"]),
              default="safetensors", help="Output format")
@click.option("--dataset-source", type=click.Choice(["builtin", "hf"]),
              default="builtin", help="Prompt dataset source")
@click.option("--max-samples", type=int, default=100,
              help="Max prompt samples for vector computation")
@click.option("--target-layer", type=int, default=-2,
              help="Layer for refusal vector computation")
@click.pass_context
def ablate(ctx, model_path, backend, output, strength, layers, targets,
           ram, vram, checkpoint_every, checkpoint_dir, validate,
           cache_vectors, output_format, dataset_source, max_samples,
           target_layer):
    """
    Apply directional ablation to remove refusal behavior from a model.

    MODEL_PATH is a HuggingFace model name or local path.

    Examples:

        unfetter ablate meta-llama/Llama-3.1-8B-Instruct

        unfetter ablate ./my-model --strength 0.8 --layers -8:-1

        unfetter ablate large-model --backend cpu --ram 16
    """
    click.echo(BANNER)
    logger = logging.getLogger("unfetter.cli")

    try:
        from unfetter.cli.commands import run_ablation

        config = {
            "model_path": model_path,
            "backend": backend,
            "output": output or f"./unfettered-{model_path.split('/')[-1]}",
            "strength": strength,
            "layers": layers,
            "targets": targets.split(","),
            "ram_limit_gb": ram,
            "vram_limit_gb": vram,
            "checkpoint_every": checkpoint_every,
            "checkpoint_dir": checkpoint_dir,
            "validate": validate,
            "cache_vectors": cache_vectors,
            "output_format": output_format,
            "dataset_source": dataset_source,
            "max_samples": max_samples,
            "target_layer": target_layer,
            "verbose": ctx.obj.get("verbose", False),
        }

        run_ablation(config)

    except Exception as e:
        logger.error(f"Ablation failed: {e}")
        if ctx.obj.get("verbose"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("checkpoint_path")
@click.pass_context
def resume(ctx, checkpoint_path):
    """Resume an interrupted ablation job from a checkpoint."""
    click.echo(BANNER)
    logger = logging.getLogger("unfetter.cli")

    try:
        from unfetter.cli.commands import run_resume
        run_resume(checkpoint_path)
    except Exception as e:
        logger.error(f"Resume failed: {e}")
        sys.exit(1)


@cli.command()
@click.argument("original_model")
@click.argument("ablated_model")
@click.option("--tests", default="refusal,helpfulness",
              help="Comma-separated tests to run")
@click.option("--output", "-o", default=None, help="Output report file")
@click.option("--max-prompts", type=int, default=50,
              help="Maximum prompts per test")
@click.pass_context
def compare(ctx, original_model, ablated_model, tests, output, max_prompts):
    """
    Compare an ablated model against the original.

    Runs refusal, helpfulness, and KL divergence tests.
    """
    click.echo(BANNER)
    logger = logging.getLogger("unfetter.cli")

    try:
        from unfetter.cli.commands import run_compare
        run_compare(original_model, ablated_model, tests.split(","), output, max_prompts)
    except Exception as e:
        logger.error(f"Compare failed: {e}")
        sys.exit(1)


@cli.command()
@click.argument("model_path")
@click.option("--tests", default="refusal,helpfulness",
              help="Comma-separated tests to run")
@click.option("--max-prompts", type=int, default=50,
              help="Maximum prompts per test")
@click.option("--output", "-o", default=None, help="Output report file")
@click.pass_context
def validate(ctx, model_path, tests, max_prompts, output):
    """Validate an ablated model's quality."""
    click.echo(BANNER)
    logger = logging.getLogger("unfetter.cli")

    try:
        from unfetter.cli.commands import run_validate
        run_validate(model_path, tests.split(","), max_prompts, output)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


@cli.command()
def info():
    """Display detected hardware and system information."""
    click.echo(BANNER)

    from unfetter.backends.auto import print_hardware_info
    click.echo(print_hardware_info())


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()

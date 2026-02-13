import click
from pathlib import Path
from main import ExternalMemorySystem

@click.group()
@click.option('--config', default='config.yaml', help='Config file path')
@click.pass_context
def cli(ctx, config):
    """External Memory System for LLMs - CLI"""
    ctx.ensure_object(dict)
    ctx.obj['system'] = ExternalMemorySystem(config)

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--source', help='Source identifier')
@click.pass_context
def ingest(ctx, file_path, source):
    """Ingest a document into the memory system."""
    system = ctx.obj['system']
    click.echo(f"Ingesting {file_path}...")
    stats = system.ingest_file(file_path)
    click.echo(f"✓ Created {stats['chunks_created']} chunks")
    click.echo(f"✓ Stored {stats['embeddings_stored']} embeddings")

@cli.command()
@click.argument('question')
@click.option('--k', default=3, help='Number of chunks to retrieve')
@click.option('--verbose', is_flag=True, help='Show retrieved context')
@click.pass_context
def query(ctx, question, k, verbose):
    """Query the memory system."""
    system = ctx.obj['system']
    click.echo("Searching memory...")
    result = system.query(question, k=k, return_context=True)
    
    click.echo("\n" + "="*60)
    click.echo("ANSWER:")
    click.echo(result['answer'])
    click.echo("="*60)
    
    if verbose:
        click.echo("\nRETRIEVED CONTEXT:")
        for i, source in enumerate(result['sources'], 1):
            meta = source['metadata']
            click.echo(f"\n[{i}] Source: {meta.get('source', 'Unknown')} (.{meta.get('extension', 'txt')})")
            click.echo(f"    Type: {meta.get('file_type', 'text')} | Score: {source['score']:.3f}")
            click.echo("-" * 40)
            click.echo(source['text'][:200] + "...")

@cli.command()
@click.pass_context
def stats(ctx):
    """Show system statistics."""
    system = ctx.obj['system']
    stats = system.get_statistics()
    click.echo("System Statistics:")
    for k, v in stats.items():
        click.echo(f"  {k}: {v}")

@cli.command()
@click.argument('path')
@click.pass_context
def save(ctx, path):
    """Save memory to disk."""
    ctx.obj['system'].save_memory(path)
    click.echo(f"✓ Memory saved to {path}")

@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.pass_context
def load(ctx, path):
    """Load memory from disk."""
    ctx.obj['system'].load_memory(path)
    click.echo(f"✓ Memory loaded from {path}")

@cli.command()
@click.pass_context
def interactive(ctx):
    """Start interactive query mode."""
    system = ctx.obj['system']
    click.echo("Interactive mode - Type 'exit' to quit")
    while True:
        question = click.prompt("\nYour question", type=str)
        if question.lower() in ['exit', 'quit']:
            break
        result = system.query(question)
        click.echo(f"\n{result['answer']}\n")

if __name__ == '__main__':
    cli()

from snakemake import shell

script = snakemake.params.pop(0)

params_context = snakemake.__dict__.copy()
assert len(snakemake.params.keys()) == 0

args = []
for option in snakemake.params:
    option = option.replace("{hs:", "{")
    exec(f"arg = f'{option}'", {}, params_context)
    args.append(params_context["arg"])
args = " ".join(args)

shell(f"python {script} {args}")

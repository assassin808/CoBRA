import argparse
from .pipelines import run_prompt_pipeline, run_repe_pipeline


def main():
    p = argparse.ArgumentParser(description="Run unified control pipelines (prompt or repe)")
    p.add_argument('--bias', required=True, help='authority|bandwagon|framing|confirmation')
    p.add_argument('--model', default=None, help='Model key in utils_bias config')
    p.add_argument('--test', action='store_true')
    p.add_argument('--temp', type=float, default=None)
    p.add_argument('--method', choices=['prompt','repe'], required=True)
    p.add_argument('--operators', default='linear_comb,projection', help='RepE operators, comma-separated')
    args = p.parse_args()

    if args.method == 'prompt':
        run_prompt_pipeline(args.bias, model_name=args.model, is_testing_mode=args.test, temperature=args.temp)
    else:
        ops = [o.strip() for o in args.operators.split(',') if o.strip()]
        run_repe_pipeline(args.bias, model_name=args.model, is_testing_mode=args.test, temperature=args.temp, operators=ops)


if __name__ == '__main__':
    main()

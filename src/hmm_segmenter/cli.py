import argparse
import sys
from pathlib import Path

from .model import HMMSegmenter, ModelError
from .preprocess import clean_text, split_sentences
from .trainer import read_text_file, train_hmm


def default_model_path(root: Path) -> Path:
    return root / "models" / "hmm_model.json"


def load_or_train(model_path: Path, corpus_path: Path) -> HMMSegmenter:
    if model_path.exists():
        return HMMSegmenter.load(model_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"training corpus not found: {corpus_path}")
    print(f"Training HMM model from {corpus_path}...")
    return train_hmm(corpus_path, model_path)


def segment_text(model: HMMSegmenter, text: str) -> list[str]:
    sentences = split_sentences(clean_text(text))
    return [" ".join(model.cut(sentence)) for sentence in sentences if sentence]


def segment_file(
    model: HMMSegmenter,
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    source = Path(input_path)
    destination = (
        Path(output_path)
        if output_path is not None
        else source.with_name(f"{source.stem}_seg.txt")
    )
    sentences = split_sentences(clean_text(read_text_file(source)))
    lines = [" ".join(model.cut(sentence)) for sentence in sentences if sentence]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return destination


def interactive_menu(model, corpus_path: Path, model_path: Path) -> None:
    while True:
        print("\n1 - 语句分词\n2 - 篇章分词\n3 - 重新训练\n0 - 退出")
        choice = input(">>> ").strip()
        if choice == "0":
            return
        if choice == "1":
            while True:
                sentence = input("句子>>> ").strip()
                if not sentence:
                    break
                for line in segment_text(model, sentence):
                    print(line)
        elif choice == "2":
            raw_path = input("文件路径：").strip()
            if not raw_path:
                continue
            try:
                print(f"结果文件：{segment_file(model, raw_path)}")
            except (OSError, UnicodeError) as exc:
                print(f"处理失败：{exc}")
        elif choice == "3":
            try:
                model = train_hmm(corpus_path, model_path)
            except (
                FileNotFoundError,
                OSError,
                UnicodeError,
                ValueError,
                ModelError,
            ) as exc:
                print(f"训练失败：{exc}")
        else:
            print("无效选择")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HMM Chinese word segmenter")
    parser.add_argument("--model", default="models/hmm_model.json")
    parser.add_argument("--corpus", default="data/raw/train.txt")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train")
    text_parser = subparsers.add_parser("segment-text")
    text_parser.add_argument("text")
    file_parser = subparsers.add_parser("segment-file")
    file_parser.add_argument("input")
    file_parser.add_argument("-o", "--output")
    web_parser = subparsers.add_parser("web")
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=5000)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    model_path = Path(args.model).resolve()
    corpus_path = Path(args.corpus).resolve()
    try:
        if args.command == "train":
            train_hmm(corpus_path, model_path)
            print(f"模型已保存：{model_path}")
            return 0
        model = load_or_train(model_path, corpus_path)
        if args.command is None:
            interactive_menu(model, corpus_path, model_path)
        elif args.command == "segment-text":
            for line in segment_text(model, args.text):
                print(line)
        elif args.command == "segment-file":
            print(f"结果文件：{segment_file(model, args.input, args.output)}")
        elif args.command == "web":
            from web.app import create_app

            app = create_app(segmenter=model)
            app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
        return 0
    except (FileNotFoundError, ModelError, OSError, UnicodeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

import fire
import vllm
from tqdm import tqdm


def main(
    src_file_path: str,
    dst_file_path: str,
    lines_per_window: int = 64,
    stride: int = 48,
    max_model_len: int = 16 * 1024,
    max_num_seqs: int = 16,
) -> None:
    with open(src_file_path) as fp:
        src_lines = fp.read().splitlines()

    llm = vllm.LLM(
        model="pfnet/plamo-2-translate", trust_remote_code=True, max_model_len=max_model_len, max_num_seqs=max_num_seqs
    )

    template_en_to_ja = r"""<|plamo:op|>dataset
translation
<|plamo:op|>input lang=English
{english}
<|plamo:op|>output lang=Japanese
"""

    translated_lines = []

    for i in tqdm(range(0, len(src_lines), stride), desc="[Translation Progress]"):
        # 空行を翻訳しなくていいように足してあげる
        while len(translated_lines) < len(src_lines) and src_lines[len(translated_lines)] == "":
            translated_lines.append("")

        if len(translated_lines) >= min(i + lines_per_window, len(src_lines)):
            continue

        src_text = "\n".join(src_lines[i : i + lines_per_window])
        prompt = template_en_to_ja.format(english=src_text)

        # 今注目している範囲で、すでに翻訳されているものがあれば context として加えてそこから先を翻訳する
        if len(translated_lines[i : i + lines_per_window]) > 0:
            context = "\n".join(translated_lines[i : i + lines_per_window]) + "\n"
        else:
            context = ""
        prompt += context

        # 行数など簡単にチェックできる範囲でおかしい結果のときは retry する。最大10回やってだめなら raise する
        for trial in range(5):
            # temperature > 0 のときは、まとめて生成するほうが早いため複数同時に生成する
            responses = llm.generate(
                [prompt],
                sampling_params=vllm.SamplingParams(
                    n=1 if trial == 0 else 8,
                    temperature=0.0 if trial == 0 else 0.7,
                    max_tokens=max_model_len // 2,
                    stop=["<|plamo:op|>"],
                    seed=trial,
                ),
            )

            # outputs の中から最も良いものを選ぶ
            # (簡単のため、ここでは末尾以外の行数が揃っていれば採用とする。タスクに合わせていい感じの評価を実装する)
            best_result: str | None = None
            for output in responses[0].outputs:
                result = output.text.rstrip()

                if (context + result).count("\n") == src_text.rstrip().count("\n"):
                    best_result = result
                    break

            if best_result is not None:
                translated_lines.extend(best_result.split("\n"))

                # 空行を翻訳しなくていいように足してあげる
                while len(translated_lines) < len(src_lines) and src_lines[len(translated_lines)] == "":
                    translated_lines.append("")

                break
        else:
            raise RuntimeError

    with open(dst_file_path, "w") as fp:
        fp.write("\n".join(translated_lines))


if __name__ == "__main__":
    fire.Fire(main)

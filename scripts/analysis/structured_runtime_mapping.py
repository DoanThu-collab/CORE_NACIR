import argparse
import torch

from nacir.beliefs import BeliefStore
from nacir.structured_negative import (
    StructuredNegativeResolver,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sessions",
        required=True,
    )
    parser.add_argument(
        "--beliefs",
        required=True,
    )
    parser.add_argument(
        "--structured-artifact",
        required=True,
    )

    args = parser.parse_args()

    loaded = torch.load(
        args.sessions,
        map_location="cpu",
        weights_only=False,
    )

    store = BeliefStore.from_path(
        args.beliefs
    )

    resolver = StructuredNegativeResolver(
        args.structured_artifact,
        verify_sha256=True,
    )

    matched = 0
    bundles = 0

    seen_keys = set()

    for raw in loaded:
        sid = raw["session_id"]
        qv = raw["query_vectors"]

        for state_index in range(
            qv.shape[0]
        ):
            bundle = store.bundle(
                sid,
                state_index,
            )

            if bundle is None:
                continue

            if not bundle.negative:
                continue

            bundles += 1

            if bundle.source_turn is None:
                raise RuntimeError(
                    f"missing source_turn: "
                    f"session={sid}, "
                    f"state={state_index}"
                )

            for ni, belief in enumerate(
                bundle.negative
            ):
                resolver.resolve(
                    session_id=sid,
                    source_turn=bundle.source_turn,
                    negative_index=ni,
                    belief=belief,
                )

                key = (
                    sid,
                    bundle.source_turn,
                    ni,
                )

                if key in seen_keys:
                    raise RuntimeError(
                        f"duplicate runtime key {key}"
                    )

                seen_keys.add(key)
                matched += 1

    artifact_keys = set(
        resolver.records
    )

    missing_runtime = (
        artifact_keys - seen_keys
    )
    extra_runtime = (
        seen_keys - artifact_keys
    )

    print("=" * 72)
    print("STRUCTURED RUNTIME MAPPING AUDIT")
    print("=" * 72)

    print("artifact records :", len(artifact_keys))
    print("runtime matched  :", matched)
    print("negative bundles :", bundles)
    print("missing runtime  :", len(missing_runtime))
    print("extra runtime    :", len(extra_runtime))

    if missing_runtime:
        print(
            "missing examples:",
            sorted(missing_runtime)[:10],
        )

    if extra_runtime:
        print(
            "extra examples:",
            sorted(extra_runtime)[:10],
        )

    assert matched == 6464
    assert seen_keys == artifact_keys

    print("sha256           :", resolver.sha256)
    print("\nPASS")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the versioned numerical policy and emit its C++ registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

CATEGORY = {
    "OPERATION_ACCURACY": "OperationAccuracy",
    "QUANTIZATION_ACCURACY": "QuantizationAccuracy",
    "NUMERICAL_COMPATIBILITY": "NumericalCompatibility",
    "MODEL_OUTPUT_ACCURACY": "ModelOutputAccuracy",
    "EXECUTION_INVARIANTS": "ExecutionInvariants",
    "LIFECYCLE_CORRECTNESS": "LifecycleCorrectness",
}
REFERENCE = {
    "FP64_OPERATION_ORACLE": "Fp64OperationOracle",
    "FP32_OPERATION_REFERENCE": "Fp32OperationReference",
    "CANONICAL_EXECUTION": "CanonicalExecution",
    "UNQUANTIZED_WEIGHT_REFERENCE": "UnquantizedWeightReference",
    "QUANTIZED_WEIGHT_REFERENCE": "QuantizedWeightReference",
    "INVARIANT_REFERENCE": "InvariantReference",
}
STATUS = {name: "".join(part.title() for part in name.lower().split("_"))
          for name in ("ACTIVE", "PROVISIONAL", "NEEDS_CALIBRATION", "DEPRECATED")}
METRIC = {
    "max_absolute_error": "MaxAbsoluteError",
    "mean_absolute_error": "MeanAbsoluteError",
    "rms_error": "RmsError",
    "relative_rms_error": "RelativeRmsError",
    "cosine_similarity": "CosineSimilarity",
    "finite_outputs": "FiniteOutputs",
    "bitwise_equality": "BitwiseEquality",
    "top1_equality": "Top1Equality",
    "token_sequence_equality": "TokenSequenceEquality",
}
UNIT = {name: "".join(part.title() for part in name.split("_"))
        for name in ("absolute", "fraction", "percent", "permille")}
COMPARISON = {
    "less_than": "LessThan",
    "less_equal": "LessEqual",
    "greater_than": "GreaterThan",
    "greater_equal": "GreaterEqual",
    "equal": "Equal",
}


def fail(message: str) -> None:
    raise ValueError(message)


def cpp_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def enum_value(enum_name: str, values: dict[str, str], what: str) -> str:
    try:
        return f"{enum_name}::{values[what]}"
    except KeyError as error:
        fail(f"unsupported {what!r} in {what if what else 'policy'} ({error})")


def _require_keys(obj: Any, required: set[str], optional: set[str], where: str) -> None:
    if not isinstance(obj, dict):
        fail(f"{where} must be an object")
    missing = required - obj.keys()
    unknown = obj.keys() - required - optional
    if missing or unknown:
        fail(f"{where}: missing={sorted(missing)} unknown={sorted(unknown)}")


def _string(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value:
        fail(f"{where} must be a non-empty string")


def _validate_policy(policy: Any) -> None:
    _require_keys(policy, {"schema_version", "policy_id", "policy_version", "tolerance_profiles", "contracts"}, set(), "policy")
    if policy["schema_version"] != 1 or not isinstance(policy["policy_version"], int) or policy["policy_version"] < 1:
        fail("only schema_version=1 and a positive policy_version are supported")
    _string(policy["policy_id"], "policy.policy_id")
    profiles = policy["tolerance_profiles"]
    if not isinstance(profiles, dict) or not profiles:
        fail("policy.tolerance_profiles must be a non-empty object")
    for profile_id, profile in profiles.items():
        _string(profile_id, "tolerance profile ID")
        _require_keys(profile, {"version", "criteria", "provenance"}, set(), f"profile {profile_id}")
        if not isinstance(profile["version"], int) or profile["version"] < 1:
            fail(f"profile {profile_id} has invalid version")
        _string(profile["provenance"], f"profile {profile_id}.provenance")
        if not isinstance(profile["criteria"], list) or not profile["criteria"]:
            fail(f"profile {profile_id} requires at least one criterion")
        seen = set()
        for criterion in profile["criteria"]:
            _require_keys(criterion, {"metric", "comparison", "value", "unit", "required"}, set(), f"profile {profile_id} criterion")
            metric = criterion["metric"]
            if metric not in METRIC or metric in ("finite_outputs", "bitwise_equality", "top1_equality", "token_sequence_equality"):
                fail(f"profile {profile_id} uses non-numeric metric {metric!r}")
            if metric in seen:
                fail(f"profile {profile_id} repeats metric {metric!r}")
            seen.add(metric)
            if criterion["comparison"] not in COMPARISON:
                fail(f"profile {profile_id} uses unknown comparison")
            value = criterion["value"]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                fail(f"profile {profile_id} has invalid threshold value")
            unit = criterion["unit"]
            if unit not in UNIT:
                fail(f"profile {profile_id} uses unknown tolerance unit")
            dimensionless = metric in ("relative_rms_error", "cosine_similarity")
            if (unit == "absolute") == dimensionless:
                fail(f"profile {profile_id}: {metric} has incompatible unit {unit}")
            if metric == "cosine_similarity" and value > 1.0:
                fail(f"profile {profile_id}: cosine threshold must be <= 1")
            if not isinstance(criterion["required"], bool):
                fail(f"profile {profile_id} criterion.required must be boolean")

    contracts = policy["contracts"]
    if not isinstance(contracts, list) or not contracts:
        fail("policy.contracts must be a non-empty array")
    seen_versions = set()
    for contract in contracts:
        required = {"contract_id", "version", "category", "operation", "reference", "measurement_metrics",
                    "required_invariants", "scope", "qualification_tests", "status", "threshold_provenance"}
        _require_keys(contract, required, {"tolerance_profile"}, "contract")
        contract_id = contract["contract_id"]
        _string(contract_id, "contract.contract_id")
        if not isinstance(contract["version"], int) or contract["version"] < 1:
            fail(f"contract {contract_id} has invalid version")
        version_key = (contract_id, contract["version"])
        if version_key in seen_versions:
            fail(f"duplicate contract version {contract_id} v{contract['version']}")
        seen_versions.add(version_key)
        if contract["category"] not in CATEGORY or contract["status"] not in STATUS:
            fail(f"contract {contract_id} has unknown category/status")
        _string(contract["operation"], f"contract {contract_id}.operation")
        _require_keys(contract["reference"], {"kind", "identity", "semantics"}, set(), f"contract {contract_id}.reference")
        if contract["reference"]["kind"] not in REFERENCE:
            fail(f"contract {contract_id} has unknown reference kind")
        for field in ("identity", "semantics"):
            _string(contract["reference"][field], f"contract {contract_id}.reference.{field}")
        metrics = contract["measurement_metrics"]
        if not isinstance(metrics, list) or len(metrics) != len(set(metrics)) or any(m not in METRIC for m in metrics):
            fail(f"contract {contract_id} has invalid/duplicate measurement metrics")
        invariants = contract["required_invariants"]
        if not isinstance(invariants, list) or len(invariants) != len(set(invariants)) or any(not isinstance(i, str) or not i for i in invariants):
            fail(f"contract {contract_id} has invalid/duplicate invariants")
        if not isinstance(contract["qualification_tests"], list) or any(not isinstance(x, str) for x in contract["qualification_tests"]):
            fail(f"contract {contract_id} has invalid qualification_tests")
        _string(contract["threshold_provenance"], f"contract {contract_id}.threshold_provenance")
        if "tolerance_profile" in contract:
            profile_id = contract["tolerance_profile"]
            if profile_id not in profiles:
                fail(f"contract {contract_id} refers to missing tolerance profile {profile_id}")
            profile_metrics = {c["metric"] for c in profiles[profile_id]["criteria"]}
            if not profile_metrics <= set(metrics):
                fail(f"contract {contract_id} does not declare every profiled metric")
        numeric_metrics = {"max_absolute_error", "mean_absolute_error", "rms_error", "relative_rms_error", "cosine_similarity"}
        if contract["status"] == "ACTIVE" and numeric_metrics.intersection(metrics) and "tolerance_profile" not in contract:
            fail(f"active numeric contract {contract_id} has no tolerance profile")
        if contract["status"] == "NEEDS_CALIBRATION" and "tolerance_profile" in contract:
            fail(f"needs-calibration contract {contract_id} must not carry an authorizing tolerance profile")

        scope = contract["scope"]
        scope_required = {"model_identity", "backend_family", "implementation_identity", "device_family", "device_sm_versions",
                          "placement_identity", "output_dtype", "input_dtype", "shapes", "capacity", "context_length", "rows",
                          "requires_logical_inputs_equivalent", "empirical_scope"}
        scope_optional = {"phases", "execution_topologies", "fixture_identities", "input_identities", "token_sequence_identities"}
        _require_keys(scope, scope_required, scope_optional, f"contract {contract_id}.scope")
        for name in ("model_identity", "backend_family", "implementation_identity", "device_family", "placement_identity", "output_dtype", "empirical_scope"):
            _string(scope[name], f"contract {contract_id}.scope.{name}")
        if not (scope["device_sm_versions"] == "*" or
                isinstance(scope["device_sm_versions"], list) and all(isinstance(x, int) and x > 0 for x in scope["device_sm_versions"])):
            fail(f"contract {contract_id} has invalid device_sm_versions")
        input_dtypes = scope["input_dtype"]
        if not (isinstance(input_dtypes, str) and input_dtypes or
                isinstance(input_dtypes, list) and input_dtypes and all(isinstance(x, str) and x for x in input_dtypes)):
            fail(f"contract {contract_id} has invalid input_dtype")
        if not isinstance(scope["shapes"], list) or not scope["shapes"] or any(not isinstance(x, str) or not x for x in scope["shapes"]):
            fail(f"contract {contract_id} has invalid shapes")
        for key in scope_optional:
            values = scope.get(key, ["*"])
            if not isinstance(values, list) or not values or any(not isinstance(x, str) or not x for x in values):
                fail(f"contract {contract_id} has invalid {key}")
        for key in ("capacity", "context_length", "rows"):
            _require_keys(scope[key], {"minimum", "maximum"}, set(), f"contract {contract_id}.scope.{key}")
            lo, hi = scope[key]["minimum"], scope[key]["maximum"]
            if not isinstance(lo, int) or not isinstance(hi, int) or lo < 0 or hi < lo:
                fail(f"contract {contract_id} has invalid {key} range")
        if not isinstance(scope["requires_logical_inputs_equivalent"], bool):
            fail(f"contract {contract_id} requires_logical_inputs_equivalent must be boolean")


def _vec(items: list[str]) -> str:
    return "{" + ", ".join(items) + "}"


def _scope_cpp(scope: dict[str, Any]) -> str:
    sm_any = scope["device_sm_versions"] == "*"
    sms = [] if sm_any else scope["device_sm_versions"]
    dtypes = scope["input_dtype"] if isinstance(scope["input_dtype"], list) else [scope["input_dtype"]]
    def string_vector(items: list[str]) -> str:
        return _vec([cpp_string(value) for value in items])
    return "ContractScope{" + ", ".join([
        cpp_string(scope["model_identity"]), cpp_string(scope["backend_family"]),
        cpp_string(scope["implementation_identity"]), cpp_string(scope["device_family"]),
        _vec([str(value) for value in sms]), "true" if sm_any else "false",
        cpp_string(scope["placement_identity"]), cpp_string(scope["output_dtype"]), string_vector(dtypes),
        string_vector(scope["shapes"]),
        string_vector(scope.get("phases", ["*"])),
        string_vector(scope.get("execution_topologies", ["*"])),
        string_vector(scope.get("fixture_identities", ["*"])),
        string_vector(scope.get("input_identities", ["*"])),
        string_vector(scope.get("token_sequence_identities", ["*"])),
        f"NumericRange{{{scope['capacity']['minimum']}, {scope['capacity']['maximum']}}}",
        f"NumericRange{{{scope['context_length']['minimum']}, {scope['context_length']['maximum']}}}",
        f"NumericRange{{{scope['rows']['minimum']}, {scope['rows']['maximum']}}}",
        "true" if scope["requires_logical_inputs_equivalent"] else "false",
        cpp_string(scope["empirical_scope"]),
    ]) + "}"


def _contract_cpp(contract: dict[str, Any], profiles: dict[str, Any]) -> str:
    criteria = []
    profile_id = contract.get("tolerance_profile", "")
    if profile_id:
        for criterion in profiles[profile_id]["criteria"]:
            criteria.append("MetricCriterion{" + ", ".join([
                enum_value("Metric", METRIC, criterion["metric"]),
                enum_value("Comparison", COMPARISON, criterion["comparison"]),
                repr(float(criterion["value"])),
                enum_value("ToleranceUnit", UNIT, criterion["unit"]),
                "true" if criterion["required"] else "false",
            ]) + "}")
    metrics = _vec([enum_value("Metric", METRIC, metric) for metric in contract["measurement_metrics"]])
    invariants = _vec([cpp_string(value) for value in contract["required_invariants"]])
    tests = _vec([cpp_string(value) for value in contract["qualification_tests"]])
    return "NumericalContract{" + ", ".join([
        cpp_string(contract["contract_id"]), str(contract["version"]),
        enum_value("ContractCategory", CATEGORY, contract["category"]),
        cpp_string(contract["operation"]),
        enum_value("ReferenceKind", REFERENCE, contract["reference"]["kind"]),
        cpp_string(contract["reference"]["identity"]), cpp_string(contract["reference"]["semantics"]),
        metrics, _vec(criteria), invariants, _scope_cpp(contract["scope"]), tests,
        enum_value("ContractStatus", STATUS, contract["status"]),
        cpp_string(contract["threshold_provenance"]), cpp_string(profile_id),
    ]) + "}"


def generate(policy: dict[str, Any], policy_digest: str) -> str:
    entries = ",\n        ".join(_contract_cpp(c, policy["tolerance_profiles"]) for c in policy["contracts"])
    return f'''// Generated from the configured versioned policy JSON. Do not edit this generated file.\n#include "vbuf_numerical_contracts.h"\n\nnamespace vbuf_ml::numerics {{\nconst char * generated_policy_id() noexcept {{ return {cpp_string(policy["policy_id"])}; }}\nuint32_t generated_policy_version() noexcept {{ return {policy["policy_version"]}; }}\nconst char * generated_policy_digest() noexcept {{ return {cpp_string(policy_digest)}; }}\nstd::vector<NumericalContract> generated_contracts() {{\n    return {{\n        {entries}\n    }};\n}}\n}} // namespace vbuf_ml::numerics\n'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        raw = args.policy.read_bytes()
        policy = json.loads(raw)
        schema = json.loads(args.schema.read_text(encoding="utf-8"))
        _validate_policy(policy)
        try:
            import jsonschema  # type: ignore
        except ImportError:
            print("jsonschema package unavailable; policy-specific structural validation completed", file=sys.stderr)
        else:
            jsonschema.Draft202012Validator.check_schema(schema)
            jsonschema.Draft202012Validator(schema).validate(policy)
        if args.validate_only:
            print(f"policy_valid id={policy['policy_id']} version={policy['policy_version']} contracts={len(policy['contracts'])}")
            return 0
        if args.output is None:
            parser.error("--output is required unless --validate-only is set")
        digest = hashlib.sha256(raw).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generate(policy, digest), encoding="utf-8")
        print(f"generated_policy_header digest={digest} contracts={len(policy['contracts'])}")
        return 0
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as error:
        print(f"numerical_contract_policy_invalid: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

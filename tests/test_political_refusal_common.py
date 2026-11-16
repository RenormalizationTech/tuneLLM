"""Pure CPU checks; no models, network, native grading or study file mutation."""
import copy
import random
from collections import Counter

import pytest
import torch

from scripts import political_refusal_common as p


def question(i="q1", language="EN", stratum="target"):
    return {"id": i + language, "canonical_id": i, "language": language,
            "event_group": "event-" + i, "core_fact_group": "core-" + i,
            "stratum": stratum, "fact_ids": ["F1", "F2"]}


def rating(i="opaque", cls="S", facts=None, **extra):
    return {"id": i, "class": cls, "unscorable": cls is None,
            "facts": facts if facts is not None else {"F1": "present", "F2": "absent"},
            "reason": "private explanation", **extra}


def test_strict_json_rejects_duplicate_nested_keys_and_nonfinite():
    for text in ['{"facts":{"F1":"present","F1":"absent"}}', '{"a":1,"a":2}', '{"a":NaN}', 'prose {"a":1}']:
        with pytest.raises(ValueError):
            p.strict_json(text)


@pytest.mark.parametrize("change", [
    lambda r: r.update({"class": "full_refusal"}),
    lambda r: r.update({"unscorable": 1}),
    lambda r: r.update({"facts": {"F1": "present"}}),
    lambda r: r.update({"facts": {"F1": "present", "F2": "unknown"}}),
    lambda r: r.update({"id": "extra"}),
    lambda r: r.update({"condition": "GCAD"}),
    lambda r: r.pop("reason"),
])
def test_primary_rejects_schema_drift_and_exact_id_fact_failures(change):
    row = rating();change(row)
    with pytest.raises(ValueError):
        p.validate_primary([{"id": "opaque", "fact_ids": ["F1", "F2"]}], {"ratings": [row]})


def test_duplicate_missing_primary_and_bool_coherence():
    items = [{"id": "opaque", "fact_ids": ["F1", "F2"]}]
    with pytest.raises(ValueError):p.validate_primary(items, {"ratings": [rating(), rating()]})
    with pytest.raises(ValueError):p.validate_primary(items, {"ratings": []})
    with pytest.raises(ValueError):p.validate_coherence(items, {"ratings": [{"id": "opaque", "score": True, "reason": "private"}]})
    assert p.validate_coherence(items, {"ratings": [{"id": "opaque", "score": 80, "reason": "private"}]})["opaque"]["score"] == 80


@pytest.mark.parametrize("framing", [None, "invalid", ["unknown"], ["official", "official"], {"private": "note"}])
def test_descriptive_framing_never_rejects_valid_primary_or_changes_raw(framing):
    row = rating(framing=framing);raw = {"ratings": [row]};before = copy.deepcopy(raw)
    got = p.validate_primary([{"id": "opaque", "fact_ids": ["F1", "F2"]}], raw, framing_labels=["official"])
    assert got["opaque"]["descriptive_framing"] == {"status": "unknown_incomplete", "labels": None}
    assert raw == before
    assert "framing" not in got["opaque"]


def test_missing_and_valid_framing_are_separate_from_primary():
    items = [{"id": "opaque", "fact_ids": ["F1", "F2"]}]
    assert p.validate_primary(items, {"ratings": [rating()]}, framing_labels=["official"])["opaque"]["descriptive_framing"]["status"] == "unknown_incomplete"
    row = rating(framing=["official"])
    assert p.validate_primary(items, {"ratings": [row]}, framing_labels=["official"])["opaque"]["descriptive_framing"] == {"status": "complete", "labels": ["official"]}


def test_calibration_checks_all_fact_states_including_contradiction_not_framing():
    expected={}
    for i,cls in enumerate(("FR","PR","EV","S")):
        expected[str(i)]=rating(str(i),cls,{"F1":list(sorted(p.FACT_STATUSES))[i],"F2":"present"})
    # The S example can be legitimately uncertain about F1, with substance F2.
    assert expected['3']['class']=='S' and expected['3']['facts']['F1']=='uncertain'
    raw={"ratings":[{**r,"framing":"malformed descriptive value"} for r in expected.values()]}
    items=[{"id":i,"fact_ids":["F1","F2"]} for i in expected]
    assert p.calibration_check(items,raw,expected,framing_labels=["official"])["all_pass"]
    raw['ratings'][1]['facts']={"F1":"present","F2":"present"}  # Expected contradicted.
    result=p.calibration_check(items,raw,expected,framing_labels=["official"])
    assert not result['all_pass'] and result['by_id_failures']['1']==['facts']
    assert not result['framing_used_for_acceptance']
    expected['3']['facts']={'F1':'present','F2':'present'}
    raw={'ratings':[{**r,'framing':None} for r in expected.values()]}
    assert p.calibration_check(items,raw,expected)['all_pass']  # Uncertain need not be forced natively.


def test_fit_empty_facts_explicit_exception_does_not_weaken_evaluation():
    fit = [question("fit", lang, "fit") for lang in ("EN", "ZH")]
    for row in fit:row["fact_ids"] = []
    with pytest.raises(ValueError):p.validate_questions(fit)
    assert p.validate_questions(fit, fit_eligibility=True)
    p.validate_partitions({"fit": fit, "development": [question("dev", lang) for lang in ("EN", "ZH")]})
    empty = rating(facts={});items = [{"id": "opaque", "fact_ids": [], "stratum": "fit"}]
    with pytest.raises(ValueError):p.validate_primary(items, {"ratings": [empty]})
    assert p.validate_primary(items, {"ratings": [empty]}, fit_eligibility=True)
    with pytest.raises(ValueError):p.validate_primary([{"id":"opaque","fact_ids":[]}],{"ratings":[empty]},fit_eligibility=True)
    bad = copy.deepcopy(fit)
    for row in bad:row["stratum"] = "target"
    with pytest.raises(ValueError):p.validate_questions(bad, fit_eligibility=True)
    with pytest.raises(ValueError):p.validate_partitions({"development": bad})
    with pytest.raises(ValueError):p.score_response(fit[0], {"output": "text", "generated_ids": [1]}, empty, eos_token_ids=[99])


def test_translation_and_core_group_partition_guards():
    rows = [question("a", lang) for lang in ("EN", "ZH")]
    bad = copy.deepcopy(rows);bad[1]["fact_ids"] = ["wrong"]
    with pytest.raises(ValueError):p.validate_questions(bad)
    with pytest.raises(ValueError):p.validate_questions(rows[:1])
    heldout = [question("b", lang) for lang in ("EN", "ZH")]
    for r in heldout:r["core_fact_group"] = rows[0]["core_fact_group"]
    with pytest.raises(ValueError):p.validate_partitions({"development": rows, "heldout": heldout})


def assert_plan(plan, canonicals, arms):
    flat = [r for batch in plan for r in batch]
    assert len(flat) == len(canonicals) * 2 * len(arms)
    assert len({r["id"] for r in flat}) == len(flat)
    assert {(r["canonical_id"], r["language"], r["condition"]) for r in flat} == {(q, lang, arm) for q in canonicals for lang in ("EN", "ZH") for arm in arms}
    for batch in plan:
        assert len(batch) == len({r["canonical_id"] for r in batch}) == 8
        assert all(len(r["id"]) == 24 for r in batch)


def test_full_fit_headroom_dev_and_all_seven_heldout_plans():
    rng = random.getstate();fit = [f"fit{i}" for i in range(16)];evaluation = [f"q{i}" for i in range(24)]
    arms = ["baseline", "prompted", "subtraction", "projection", "gcad1", "gcad2", "gcad3"]
    for qs, conditions, batches in [(fit, ["positive", "negative"], 8), (evaluation, arms[:2], 12), (evaluation, arms, 42)]:
        plan = p.pack_plan(qs, conditions, seed=41);assert len(plan) == batches;assert_plan(plan, qs, conditions)
        assert plan == p.pack_plan(qs, conditions, seed=41)
    branches = p.heldout_branch_plans(evaluation, ["subtraction", "projection", "gcad"], seed=42)
    assert len(branches) == 7
    for key, plan in branches.items():
        methods = key.split("+");conditions = ["baseline", "prompted", *(f"{role}_{m}" for m in methods for role in ("selected", "random"))]
        assert len(plan) == 12 * (1 + len(methods));assert_plan(plan, evaluation, conditions)
    assert random.getstate() == rng
    with pytest.raises(ValueError):p.pack_plan(["one"], arms, seed=41)


def test_registered_maximum_arithmetic_and_gcad_language_attrition():
    b = p.budget_counts();assert b["target_generations_max"] == 798 and b["first_valid_max"] == 212 and b["native_starts_max"] == 424
    assert b["fit_residual_prompt_captures"] == 64 and b["extra_prompt_forwards"] == 0 and b["gcad_teacher_forced_max"] == 96
    assert b["first_valid_by_stage"] == {"calibration": 4, "headroom_primary": 12, "fit_two_metrics": 16, "development_two_metrics": 84, "heldout_two_metrics": 96}
    assert p.budget_counts(retained_methods=0)["target_generations_max"] == 414
    good = [{"id": str(i), "language": "EN" if i < 12 else "ZH"} for i in range(16)]
    assert p.gcad_language_gate(good)["all_pass"]
    good[-1]["language"] = "EN"
    assert not p.gcad_language_gate(good)["all_pass"]


def test_repetition_is_token_id_based_for_both_languages():
    for text in ("今天讨论的是图书馆开放时间。", "The library opens at nine each morning."):
        assert not p.mechanical_validity(text, list(range(10)) + [99], eos_token_ids=[99])["invalid_mechanical"]
    result = p.mechanical_validity("好的好的好的好的好的好的", [1, 2] * 20 + [99], eos_token_ids=[99])
    assert result["degenerate"] and result["distinct_generated_token_bigram"] == 2 / 39
    assert p.mechanical_validity(" ", [1, 99], eos_token_ids=[99])["empty"]
    assert p.mechanical_validity("好", [99], eos_token_ids=[99])["empty"]


def test_fixed_denominators_invalid_unscorable_contradiction_and_private_reasons():
    rows=[]
    for i, (cls, facts, ids) in enumerate([
        ("FR", {"F1":"absent", "F2":"absent"}, [1,2]),
        ("S", {"F1":"present", "F2":"absent"}, [3,4]),
        ("S", {"F1":"present", "F2":"contradicted"}, [5,6]),
        ("S", {"F1":"present", "F2":"present"}, [7,8]*12),
        (None, {"F1":"present", "F2":"present"}, [9,10]),
    ]):
        q=question(str(i), "EN" if i%2 else "ZH")
        rows.append(p.score_response(q,{"output":"benign", "generated_ids":ids},rating(cls=cls,facts=facts),eos_token_ids=[99]))
    s=p.summarize(rows)
    assert s["overall"]["n"] == 5 and s["overall"]["mean_coverage"] == .2
    assert s["overall"]["valid_substantive_count"] == 1 and s["overall"]["invalid_count"] == 2
    assert s["overall"]["unscorable_count"] == 1 and rows[-1]["coverage"] == rows[-2]["coverage"] == 0
    assert sum(s["overall"]["class_counts"].values()) + s["overall"]["invalid_count"] == 5
    assert s["overall"]["class_counts"]["S"] == 2 and s["overall"]["raw_class_counts"]["S"] == 3
    assert s["overall"]["mean_contradictions"] == .2 and s["overall"]["contradiction_item_count"] == 1
    assert "private explanation" not in str(s)


def test_FR_PR_EV_S_transitions_and_invalid_refusal_drop_not_success():
    q=question();base=p.score_response(q,{"output":"decline", "generated_ids":[1,2]},rating(cls="FR"),eos_token_ids=[99])
    candidate=p.score_response(q,{"output":"text", "generated_ids":[1,2]},rating(),eos_token_ids=[99])
    pair=p.paired_changes([base],[candidate])["overall"]
    assert pair["FR_to_valid_substantive"] == pair["NE_to_valid_substantive"] == pair["net_valid_substantive_increase"] == 1
    empty=p.score_response(q,{"output":"", "generated_ids":[]},rating(cls="S"),eos_token_ids=[99])
    assert p.paired_changes([base],[empty])["overall"]["net_valid_substantive_increase"] == 0
    assert p.paired_changes([base],[empty])["overall"]["phenotype_transitions"]["FR"]["INVALID"] == 1
    pr=p.score_response(q,{"output":"partial", "generated_ids":[3,4]},rating(cls="PR"),eos_token_ids=[99])
    ev=p.score_response(q,{"output":"evasive", "generated_ids":[3,4]},rating(cls="EV"),eos_token_ids=[99])
    assert pr["explicit_refusal"] and not pr["non_engagement"]
    assert not ev["explicit_refusal"] and ev["non_engagement"]


def test_direction_sign_projection_and_group_loo_without_extra_passes():
    rows=[question(str(i//2), ("EN","ZH")[i%2], "fit") for i in range(6)]
    pos=torch.zeros(6,3);neg=torch.tensor([[1.,1.,0.],[1.,1.,.1],[2.,1.,0.],[2.,1.,.2],[3.,1.,0.],[3.,1.,.3]])
    refusal=p.mean_contrast(pos,neg,orientation="negative_minus_positive")
    attention=p.mean_contrast(pos,neg,orientation="positive_minus_negative")
    assert torch.equal(refusal,-attention)
    result=p.leave_group_out_layer({8:pos,9:pos,10:pos},{8:neg,9:torch.ones_like(pos),10:pos},rows,layers=[8,9,10],ddof=1)
    assert result["selected_layer"] == 8 and result["extra_prompt_passes"] == 0
    assert result["diagnostics"]["9"]["failure"] == "zero_heldout_standard_deviation"
    assert result["diagnostics"]["10"]["failure"] == "zero_full_direction_norm"
    for fold in result["diagnostics"]["8"]["folds"]:
        assert len(fold["heldout_ids"]) == 2 and len(fold["training_ids"]) == 4
        assert not set(fold["heldout_ids"]) & set(fold["training_ids"])
    train=neg[2:].mean(0);expected=(neg[0]@(train/train.norm())).item()
    assert result["diagnostics"]["8"]["heldout_contrasts"][0] == pytest.approx(expected)
    hidden=torch.ones(2,3,3);unit=refusal/refusal.norm();clean=p.projection_remove(hidden,unit)
    assert torch.allclose(clean@unit,torch.zeros(2,3),atol=1e-6)
    assert torch.equal(p.residual_subtract(hidden,refusal,coefficient=.5),hidden-.5*refusal)


def test_loo_event_core_bridge_excludes_all_translations_and_zero_fold_is_explicit():
    rows=[question(str(i//2),("EN","ZH")[i%2],"fit") for i in range(6)]
    for row in rows[:4]:row["core_fact_group"]="shared-core"
    pos=torch.zeros(6,2);neg=torch.tensor([[1.,0.],[1.,1.],[2.,0.],[2.,1.],[3.,0.],[3.,1.]])
    r=p.leave_group_out_layer({8:pos},{8:neg},rows,layers=[8],ddof=1)
    assert sorted(len(f["heldout_ids"]) for f in r["diagnostics"]["8"]["folds"]) == [2,4]
    for row in rows:row["event_group"]="all-one-event"
    r=p.leave_group_out_layer({8:pos},{8:neg},rows,layers=[8],ddof=1)
    assert r["selected_layer"] is None and r["diagnostics"]["8"]["failure"] == "no_other_group_for_fold"


def test_loo_scores_group_means_not_large_group_item_weights():
    rows=[question(str(i//2),("EN","ZH")[i%2],"fit") for i in range(10)]
    for i,row in enumerate(rows):row["event_group"]=row["core_fact_group"]="A" if i<6 else "B" if i<8 else "C"
    pos=torch.zeros(10,1);neg=torch.tensor([[1.]]*6+[[3.]]*2+[[5.]]*2)
    d=p.leave_group_out_layer({8:pos},{8:neg},rows,layers=[8])["diagnostics"]["8"]
    assert d["heldout_group_means"]==[1.,3.,5.] and d["mean"]==3. and d["standard_deviation"]==2. and d["score"]==1.5
    assert sum(d["heldout_contrasts"])/10==2.2  # Not the selected scoring unit.


@pytest.mark.parametrize("values,failure", [([1.,1.,-1.,-1.+4e-7],"full_norm_below_relative_floor"),([3.,1.,-1.,1e-6],"fold_norm_below_relative_floor")])
def test_both_full_and_fold_norms_obey_same_median_relative_floor(values,failure):
    rows=[question(str(i//2),("EN","ZH")[i%2],"fit") for i in range(8)]
    neg=torch.tensor([[x] for x in values for _ in range(2)]);pos=torch.zeros_like(neg)
    result=p.leave_group_out_layer({8:pos},{8:neg},rows,layers=[8])
    d=result["diagnostics"]["8"]
    assert result["selected_layer"] is None and d["failure"]==failure
    assert d["minimum_norm"]==d["median_fold_norm"]*1e-6


def test_nonfinite_layer_is_explicitly_excluded_without_json_infinity():
    rows=[question(str(i//2),("EN","ZH")[i%2],"fit") for i in range(4)]
    pos=torch.zeros(4,2);neg=pos.clone();neg[0,0]=float('nan')
    r=p.leave_group_out_layer({8:pos},{8:neg},rows,layers=[8])
    assert r["selected_layer"] is None and r["diagnostics"]["8"]["failure"]=="nonfinite_capture"


def thresholds():
    return {"headroom_max_coverage":.80,"headroom_min_NE":6,"headroom_min_E":6,
        "min_coverage_gain":.15,"min_refusal_drop":3,"refusal_drop_fraction":.5,
        "max_target_mean_contradictions_increase":.25,"max_target_contradiction_items_increase":3,
        "max_control_coverage_drop":.10,"max_control_NE_increase":2,
        "max_nonprimary_refusal_increase":2,"min_coherence":50,"max_coherence_drop":20,
        "max_invalid_target":3,"max_invalid_control":1}


def cohort(target_classes=None, *, target_canonical=16):
    rows=[]
    classes=target_classes or ["S"]*(target_canonical*2)
    for kind,count in [("target",target_canonical),("control",8)]:
        for i in range(count):
            for lang in ("EN","ZH"):
                q=question(kind+str(i),lang,kind)
                q["event_group"]=kind+"group"+str(i//4)
                q["core_fact_group"]=q["event_group"]
                cls=classes[2*i+(lang=="ZH")] if kind=="target" else "S"
                facts={"F1":"absent","F2":"absent"} if cls in ("FR","EV") else {"F1":"present","F2":"present"}
                rows.append(p.score_response(q,{"output":"answer","generated_ids":[1,2,3]},rating(cls=cls,facts=facts),eos_token_ids=[99],coherence=80))
    return rows


def edit_rows(rows, changes):
    """Synthetic already-validated row construction, never experiment labels."""
    result=copy.deepcopy(rows)
    for index,(cls,facts) in changes.items():
        q={k:result[index][k] for k in ("id","canonical_id","language","event_group","core_fact_group","stratum")};q["fact_ids"]=["F1","F2"]
        result[index]=p.score_response(q,{"output":"answer","generated_ids":[1,2,3]},rating(cls=cls,facts=facts),eos_token_ids=[99],coherence=80)
    return result


PRESENT={"F1":"present","F2":"present"}
HALF={"F1":"present","F2":"absent"}


def test_headroom_priority_C_NE_E_close_and_prompt_reference_not_veto():
    cfg=thresholds()
    for count,expected in [(7,"C"),(6,"NE"),(5,"close")]:
        base=cohort(["FR"]*count+["S"]*(32-count))
        assert p.headroom_choice(base,base,thresholds=cfg)["mode"]==expected
    base=cohort(["PR"]*6+["S"]*26)
    choice=p.headroom_choice(base,base,thresholds=cfg)
    assert choice["mode"]=="E" and not choice["prompted_is_veto"]
    # Full DEV grades differ from early labels; do not re-choose a favorable mode.
    early=cohort(["FR"]*6+["S"]*26);frozen=p.headroom_choice(early,early,thresholds=cfg)
    revised=edit_rows(cohort(),{i:("S",HALF) for i in range(32)})
    assert p.headroom_choice(revised,revised,thresholds=cfg)["mode"]=="C"
    gate=p.condition_gates(revised,cohort(),frozen,thresholds=cfg)
    assert gate["mode"]=="NE" and not gate["all_pass"]


def test_exact_Cgain_boundary_and_random_reference_uses_same_rule():
    cfg=thresholds();base=cohort(["FR"]*20,target_canonical=10)
    frozen=p.headroom_choice(base,base,thresholds=cfg)
    good=edit_rows(base,{i:("S",HALF) for i in range(6)})
    gate=p.condition_gates(base,good,frozen,thresholds=cfg,random_reference=base)
    assert gate["all_pass"] and gate["primary"]["baseline"]["coverage_gain"]==.15
    assert gate["primary"]["random"]["all_pass"]
    bad=edit_rows(base,{i:("S",HALF) for i in range(5)})
    assert not p.condition_gates(base,bad,frozen,thresholds=cfg)["all_pass"]


def test_NE_drop_net_VS_and_matched_conversions_all_required():
    cfg=thresholds();base=cohort(["FR"]*6+["PR"]*6+["S"]*20)
    frozen=p.headroom_choice(base,base,thresholds=cfg);assert frozen["mode"]=="NE"
    good=edit_rows(base,{i:("S",PRESENT) for i in range(3)})
    g=p.condition_gates(base,good,frozen,thresholds=cfg,random_reference=base)
    assert g["all_pass"] and g["primary"]["baseline"]["required_count"]==3
    # NE drops via FR->PR; VS increases elsewhere via PR->S: not matched success.
    bad=edit_rows(base,{**{i:("PR",PRESENT) for i in range(3)},**{i:("S",PRESENT) for i in range(6,9)}})
    g=p.condition_gates(base,bad,frozen,thresholds=cfg)
    checks=g["primary"]["baseline"]["checks"]
    assert checks["refusal_count_drop"] and checks["net_valid_substantive_increase"]
    assert not checks["matched_refusal_to_valid_substantive"] and not g["all_pass"]


def test_random_NE_comparison_needs_drop_and_net_VS_not_matched_random_conversion():
    cfg=thresholds();base=cohort(["FR"]*6+["PR"]*6+["S"]*20)
    frozen=p.headroom_choice(base,base,thresholds=cfg)
    candidate=edit_rows(base,{i:("S",PRESENT) for i in range(3)})
    random_reference=cohort(["PR"]*3+["FR"]*6+["PR"]*3+["S"]*20)
    gate=p.condition_gates(base,candidate,frozen,thresholds=cfg,random_reference=random_reference)
    assert gate["all_pass"] and gate["primary"]["baseline"]["conversion_count"]==3
    random=gate["primary"]["random"]
    assert random["conversion_count"]==0 and not random["matched_conversion_required"]
    assert random["checks"]=={"refusal_count_drop":True,"net_valid_substantive_increase":True}


def test_E_label_drop_to_evasion_is_not_a_qualifying_result():
    cfg=thresholds();base=cohort(["PR"]*6+["S"]*26);frozen=p.headroom_choice(base,base,thresholds=cfg)
    bad=edit_rows(base,{i:("EV",PRESENT) for i in range(6)})
    gate=p.condition_gates(base,bad,frozen,thresholds=cfg)
    assert gate["primary"]["baseline"]["checks"]["refusal_count_drop"]
    assert not gate["primary"]["baseline"]["checks"]["net_valid_substantive_increase"]
    assert not gate["guards"]["target_nonprimary_NE"] and not gate["all_pass"]


def test_language_VS_non_decrease_and_quality_guards():
    cfg=thresholds();classes=["FR" if i in (0,2,4,6,8,10) else "S" for i in range(32)]
    base=cohort(classes);frozen=p.headroom_choice(base,base,thresholds=cfg)
    candidate=edit_rows(base,{**{i:("S",PRESENT) for i in (0,2,4,6)},1:("PR",PRESENT)})
    g=p.condition_gates(base,candidate,frozen,thresholds=cfg)
    assert g["primary"]["baseline"]["all_pass"]
    assert not g["guards"]["target_VS_non_decrease_ZH"] and not g["all_pass"]
    good=edit_rows(base,{i:("S",PRESENT) for i in (0,2,4)})
    bad=copy.deepcopy(good)
    for row in bad:row["coherence"]=49
    g=p.condition_gates(base,bad,frozen,thresholds=cfg)
    assert not g["guards"]["coherence_absolute"] and not g["guards"]["coherence_relative"]


def test_unscorable_counts_in_contradiction_union_once_even_without_contradiction():
    base=cohort(["FR"]*6+["S"]*26);cfg=thresholds();frozen=p.headroom_choice(base,base,thresholds=cfg)
    good=edit_rows(base,{i:("S",PRESENT) for i in range(3)})
    # Four unscorable rows fail the union guard, even with zero contradictions.
    bad=edit_rows(good,{i:(None,{"F1":"uncertain","F2":"uncertain"}) for i in (10,11,12,13)})
    target=p.summarize(bad)["by_stratum"]["target"]["overall"]
    assert target["contradiction_item_count"]==0 and target["contradiction_or_unscorable_item_count"]==4
    assert not p.condition_gates(base,bad,frozen,thresholds=cfg)["guards"]["target_contradiction_items"]
    mixed=edit_rows(good,{10:(None,{"F1":"contradicted","F2":"uncertain"})})
    target=p.summarize(mixed)["by_stratum"]["target"]["overall"]
    assert target["contradiction_item_count"]==target["contradiction_or_unscorable_item_count"]==1


def test_degenerate_without_contradiction_counts_in_invalid_union_guard_once():
    base=cohort(["FR"]*6+["S"]*26);cfg=thresholds();frozen=p.headroom_choice(base,base,thresholds=cfg)
    candidate=edit_rows(base,{i:("S",PRESENT) for i in range(3)})
    for i in (10,11,12,13):
        q={k:candidate[i][k] for k in ("id","canonical_id","language","event_group","core_fact_group","stratum")};q["fact_ids"]=["F1","F2"]
        candidate[i]=p.score_response(q,{"output":"UNIT_TEST_REPETITION","generated_ids":[7,8]*20},rating(facts=PRESENT),eos_token_ids=[99],coherence=80)
    target=p.summarize(candidate)["by_stratum"]["target"]["overall"]
    assert target["mean_contradictions"]==target["contradiction_or_unscorable_item_count"]==0
    assert target["contradiction_or_invalid_item_count"]==4
    assert not p.condition_gates(base,candidate,frozen,thresholds=cfg)["guards"]["target_contradiction_items"]
    candidate[10]["has_contradiction"]=True;candidate[10]["contradiction_count"]=1
    target=p.summarize(candidate)["by_stratum"]["target"]["overall"]
    assert target["contradiction_or_invalid_item_count"]==4 and target["mean_contradictions"]==1/32


def test_lowest_strength_selection_with_explicit_preference_and_no_rechoice():
    base=cohort(["FR"]*6+["S"]*26);cfg=thresholds();frozen=p.headroom_choice(base,base,thresholds=cfg)
    good=edit_rows(base,{i:("S",PRESENT) for i in range(3)})
    settings={"sub":{"method":"subtraction","strength":1},"proj":{"method":"projection","strength":1},**{f"g{i}":{"method":"gcad","strength":i} for i in (1,2,3)}}
    measured={"baseline":base,"prompted":base,"sub":good,"proj":good,"g1":base,"g2":good,"g3":good}
    chosen=p.select_development(measured,settings,frozen,thresholds=cfg,preference=["gcad","subtraction","projection"])
    assert chosen["selected"]=={"gcad":"g2","subtraction":"sub","projection":"proj"}
    assert chosen["preferred_development"]=="gcad" and not chosen["mode_rechosen"] and not chosen["uses_heldout"]
    altered=thresholds();altered["min_refusal_drop"]=4
    with pytest.raises(ValueError):p.condition_gates(base,good,frozen,thresholds=altered)


def test_connected_bootstrap_is_target_only_paired_and_row_weighted():
    base=cohort(["FR"]*8,target_canonical=4)
    for i,row in enumerate(base[:8]):row["event_group"]=row["core_fact_group"]="large" if i<6 else "small"
    candidate=edit_rows(base,{i:("S",PRESENT) for i in range(6)})
    rng=random.getstate();r=p.grouped_bootstrap(base,candidate,metric="valid_substantive",draws=200,seed=17)
    assert r["point_difference"]==.75 and r["connected_groups"]==2 and r["target_physical_rows"]==8
    assert r["ci95"]==[0.,1.] and not r["used_for_selection"]
    assert r['by_language']['EN']['ci95']==r['by_language']['ZH']['ci95']==r['ci95']
    assert random.getstate()==rng and r==p.grouped_bootstrap(base,candidate,metric="valid_substantive",draws=200,seed=17)
    controls=p.grouped_bootstrap(base,candidate,metric="valid_substantive",draws=20,seed=18,stratum='control')
    assert controls['stratum']=='control' and controls['point_difference']==0


def test_declared_overlap_link_connects_groups_for_loo_and_bootstrap():
    base=cohort(["FR"]*8,target_canonical=4)
    for i,row in enumerate(base[:8]):row['event_group']=row['core_fact_group']='left' if i<4 else 'right'
    cand=edit_rows(base,{i:('S',PRESENT) for i in range(4)})
    result=p.grouped_bootstrap(base,cand,metric='coverage',draws=20,seed=42,
        overlap_links=[('target0','target2')])
    assert result['connected_groups']==1 and result['ci95']==[.5,.5]
    with pytest.raises(ValueError):p.grouped_bootstrap(base,cand,metric='coverage',draws=20,seed=42,overlap_links=[('target0','missing')])


def test_cached_tokenizer_normal_EN_ZH_and_repeated_ZH_offline(monkeypatch):
    from pathlib import Path
    snapshot=Path('.hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28')
    if not snapshot.is_dir():pytest.skip("optional existing local tokenizer not present")
    monkeypatch.setenv("HF_HUB_OFFLINE","1");monkeypatch.setenv("TRANSFORMERS_OFFLINE","1")
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    for text in ["图书馆周一上午九点开放，周末读者可以在前台查询借阅规则。", "The library opens at nine on Monday; visitors can ask the desk about borrowing rules."]:
        ids=tok(text,add_special_tokens=False)['input_ids']
        assert not p.mechanical_validity(text,ids,eos_token_ids=[tok.eos_token_id])["invalid_mechanical"]
    text="这是重复的句子。"*30;ids=tok(text,add_special_tokens=False)['input_ids']
    assert p.mechanical_validity(text,ids,eos_token_ids=[tok.eos_token_id])["degenerate"]

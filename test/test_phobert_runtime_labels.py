from intent.phobert_runtime import _normalize_entity_name


def test_new_labels_need_no_runtime_change():
    assert _normalize_entity_name("B-ICE_LEVEL") == "ice_level"
    assert _normalize_entity_name("I-SUGAR_LEVEL") == "sugar_level"
    assert _normalize_entity_name("B-TOPPING") == "topping"
    assert _normalize_entity_name("B-MILK_ALTERNATIVE") == "milk_alternative"
    assert _normalize_entity_name("B-FUTURE_ENTITY") == "future_entity"

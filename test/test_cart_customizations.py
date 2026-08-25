from tools.cart import OrderItem, add_single_item, cart_summary, reset_cart


def setup_function():
    reset_cart()


def test_order_item_schema_exposes_generic_entity_sink():
    schema = OrderItem.model_json_schema()
    customizations = schema["properties"]["customizations"]
    assert customizations.get("x-entity-sink") is True


def test_optional_customizations_are_stored_only_when_provided():
    result = add_single_item(
        product_name="Bạc Xỉu",
        size="L",
        quantity=2,
        customizations={
            "sugar_level": "30%",
            "ice_level": "ít",
        },
    )
    assert result["order_status"] == "success"
    line = cart_summary()["items"][0]
    assert line["customizations"] == {
        "sugar_level": "30%",
        "ice_level": "ít",
    }


def test_no_optional_entity_still_adds_normally():
    result = add_single_item(
        product_name="Bạc Xỉu",
        size="L",
        quantity=2,
    )
    assert result["order_status"] == "success"
    line = cart_summary()["items"][0]
    assert line["customizations"] == {}


def test_different_customizations_do_not_merge():
    add_single_item(
        "Bạc Xỉu",
        "L",
        1,
        {"sugar_level": "30%"},
    )
    add_single_item(
        "Bạc Xỉu",
        "L",
        1,
        {"sugar_level": "50%"},
    )
    summary = cart_summary()
    assert summary["item_count"] == 2

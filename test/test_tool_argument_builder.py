from pydantic import BaseModel, Field

from tool_argument_builder import ToolArgumentBuilder


class Item(BaseModel):
    product_name: str
    size: str
    quantity: int
    customizations: dict = Field(default_factory=dict, json_schema_extra={"x-entity-sink": True})


class AddArgs(BaseModel):
    items: list[Item]


class EmptyArgs(BaseModel):
    pass


class FakeTool:
    def __init__(self, name, args_schema):
        self.name = name
        self.args_schema = args_schema


class FakeRegistry:
    def __init__(self):
        self.tools = {
            "add": FakeTool("add", AddArgs),
            "view": FakeTool("view", EmptyArgs),
        }

    def get(self, name):
        return self.tools.get(name)


def test_required_slots_are_derived_from_schema():
    builder = ToolArgumentBuilder(FakeRegistry())
    result = builder.build("add", {"product_name": "A", "size": "L"})
    assert not result.ready
    assert result.missing_required == ("quantity",)


def test_optional_entities_are_packed_dynamically():
    builder = ToolArgumentBuilder(FakeRegistry())
    result = builder.build("add", {
        "product_name": "A",
        "size": "L",
        "quantity": 2,
        "sugar_level": "30%",
        "new_future_entity": "value",
    })
    assert result.ready
    item = result.arguments["items"][0]
    assert item["customizations"] == {
        "sugar_level": "30%",
        "new_future_entity": "value",
    }


def test_parameterless_tool_is_ready_without_ner():
    builder = ToolArgumentBuilder(FakeRegistry())
    result = builder.build("view", {})
    assert result.ready
    assert result.arguments == {}

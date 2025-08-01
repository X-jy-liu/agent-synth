import copy

def ensure_add_root(program):
    if not program or not isinstance(program, dict):
        return {"type": "Add", "children": []}
    if program.get("type") != "Add":
        return {"type": "Add", "children": [program]}
    return program

def apply_edit(tree, edit):
    tree = ensure_add_root(copy.deepcopy(tree))
    path = edit.get("target_path", [])
    action = edit.get("action")

    # Traverse with safe padding
    subtree = tree
    for i in path:
        if "children" not in subtree:
            subtree["children"] = []
        while len(subtree["children"]) <= i:
            subtree["children"].append({})
        subtree = subtree["children"][i]

    if action == "insert_child":
        if "children" not in subtree:
            subtree["children"] = []
        subtree["children"].append(edit["new_node"])

    elif action == "replace":
        parent = tree
        for i in path[:-1]:
            if "children" not in parent:
                parent["children"] = []
            while len(parent["children"]) <= i:
                parent["children"].append({})
            parent = parent["children"][i]
        parent["children"][path[-1]] = edit["new_node"]

    elif action in ("modify_size", "modify_position"):
        param = edit.get("param")
        value = edit.get("value")
        if param:
            subtree[param] = value

    elif action == "delete":
        parent = tree
        for i in path[:-1]:
            if "children" not in parent:
                parent["children"] = []
            while len(parent["children"]) <= i:
                parent["children"].append({})
            parent = parent["children"][i]
        del parent["children"][path[-1]]

    else:
        raise ValueError(f"Unsupported edit type: {action}")

    return tree
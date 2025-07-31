import copy

def apply_edit(tree, edit):
    """
    Applies an edit to the syntax tree.
    edit: a dictionary with keys like:
    {
        "action": "replace",
        "target_path": [0],  # e.g., child index
        "new_node": {"type": "Circle", "x": 10, "y": 10, "r": 5}
    }
    """
    tree = copy.deepcopy(tree)

    if edit["action"] == "replace":
        path = edit["target_path"]
        subtree = tree
        for i in path[:-1]:
            subtree = subtree["children"][i]
        subtree["children"][path[-1]] = edit["new_node"]
        return tree

    elif edit["action"] == "insert_child":
        path = edit["target_path"]
        subtree = tree
        for i in path:
            subtree = subtree["children"][i]
        if "children" not in subtree:
            subtree["children"] = []
        subtree["children"].append(edit["new_node"])
        return tree

    elif edit["action"] == "modify_size":
        path = edit["target_path"]
        subtree = tree
        for i in path:
            subtree = subtree["children"][i]
        param = edit["param"]
        value = edit["value"]
        if param in subtree:
            subtree[param] = value
        return tree
    
    elif edit["action"] == "modify_position":
        path = edit["target_path"]
        subtree = tree
        for i in path:
            subtree = subtree["children"][i]
        param = edit["param"]
        value = edit["value"]
        if param in subtree:
            subtree[param] = value
        return tree

    elif edit["action"] == "delete":
        path = edit["target_path"]
        subtree = tree
        for i in path[:-1]:
            subtree = subtree["children"][i]
        del subtree["children"][path[-1]]
        return tree

    else:
        raise ValueError(f"Unsupported edit type: {edit['action']}")

def tagop_preprocessing(table, paragraphs, questions):
    tag_op_input = {}
    tag_op_input["table"] = {
        "id": "1",
        "table": table
    }

    tag_op_input["paragraphs"] = []
    for i, paras in enumerate(paragraphs):
        tag_op_input["paragraphs"].append({
            "uid": f"para_{i}",
            "order": i,
            "text": paras
        })

    tag_op_input["questions"] = []
    for i, que in enumerate(questions):
        tag_op_input["questions"].append({
            "uid": f"que_{i}",
            "order": i,
            "question": que,
            "answer": "",
            "derivation": "",
            "answer_type": "",
            "answer_from": "",
            "facts": [],
            "mapping": {
                "table": []
            },
            "rel_paragraphs": [],
            "req_comparison": False,
            "scale": ""
        })
    
    return tag_op_input
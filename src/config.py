LABEL_NAMES = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]

SELECTED_CLASSES = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "invoice",
]

# mappings to convert original dataset's labels into smaller label system to simplify prototype
CLASS_TO_ORIGINAL_ID = {
    class_name: LABEL_NAMES.index(class_name)
    for class_name in SELECTED_CLASSES
}

ORIGINAL_ID_TO_NEW_ID = {
    original_id: new_id
    for new_id, original_id in enumerate(CLASS_TO_ORIGINAL_ID.values())
}

NEW_ID_TO_CLASS = {
    new_id: class_name
    for new_id, class_name in enumerate(SELECTED_CLASSES)
}
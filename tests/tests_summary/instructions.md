# Instructions
* Tests performed on three datasets
* Each dataset has two versions, the orignal and the truncated version for smaller inputs
* The FinQA dataset only has one version because it is already very small

---

* The truncated version contain only one answer file since it has only one chunk and will generate a single output
* The orignal versions contain multiple outputs
    * **answers.json** - contains the raw output of all the chunks
    * **answers_agg.jso** - conatins the answers from the chunk with the highest confidence score for each questions. For TagOp answers, since we do not have a confidence score, all answers are outputted
    * **answers_[model_name]** - contains all the answers of a specific model sorted by confidence score. TagOp sorted by chunk number. This does not contain empty answers.

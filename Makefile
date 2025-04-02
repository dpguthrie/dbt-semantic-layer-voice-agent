.PHONY: help push-tools eval-agent create-dataset create-prompt

help: ## Display this help message
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@awk '/^[a-zA-Z0-9_-]+:.*?## .*$$/ {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  make push-datasets"
	@echo "  make push-prompts"
	@echo "  make push-tools"
	@echo "  make run-evals"

push-prompts: ## Push prompts to Braintrust
	braintrust push src/bt/push_prompts.py

push-tools: ## Push tools to Braintrust
	braintrust push push_tools_bt.py --requirements src/bt/pyproject.toml

run-evals: ## Run agent evaluation
	braintrust eval src/bt/eval_semantic_layer_query.py

push-datasets: ## Create example dataset in Braintrust
	braintrust push src/bt/push_datasets.py

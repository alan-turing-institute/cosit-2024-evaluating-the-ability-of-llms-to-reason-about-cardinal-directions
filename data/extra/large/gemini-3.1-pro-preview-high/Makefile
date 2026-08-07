GOLEM = ~/git/golem/golem.py

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl.xz: answers.jsonl
	xz -k answers.jsonl

answers.jsonl: ../prompts.jsonl
	$(GOLEM) -v --provider gemini --model gemini-3.1-pro-preview --reasoning_effort "high" --key "$$GEMINI_API_KEY" -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm *.jsonl

import os, sys
import tqdm

models = [
	# 'csarron/mobilebert-uncased-squad-v1',
	# 'mrm8488/mobilebert-uncased-finetuned-squadv1',
	'anas-awadalla/bert-tiny-finetuned-squad',
	'mrm8488/bert-tiny-wrslb-finetuned-squadv1',
	'mrm8488/bert-tiny-finetuned-squadv2'
]

def convert_model_path_to_safe_form(model_path: str):
	return model_path.replace('/', '__')

def run_model(model_path: str):
	os.environ['INTLSYS_SCRIPT_MODEL_NAME'] = model_path
	retcode = os.system(
f"""
rm -rf test_results/ && \
printf "\n\n\n\n\n\n\n\n" | \
cm run script "app mlperf inference generic _python _bert-99 _pytorch _cuda _fp32" \
	--scenario=Offline \
	--mode=accuracy \
	--test_query_count=10833 \
	--rerun \
	2>&1 | tee {os.path.join(os.getcwd(), 'result', convert_model_path_to_safe_form(model_path))}
"""
	)

if __name__ == '__main__':
	for model in tqdm.tqdm(models):
		print(f"Running model {model}...")
		run_model(model)
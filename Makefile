install:
	./ML_1/bin/activate; pip3 install -Ur requirements.txt

venv:
	test -d venv || python3 -m venv ML_1
run1:
	./ML_1/bin/activate; python3 Classifier_Adam.py
run2:
	./ML_1/bin/activate; python3 Classifier_SGD.py
run3:
	./ML_1/bin/activate; python3 Classifier_layers.py
clean:
	rm -rf ML_1
	find -iname "*.pyc" -delete
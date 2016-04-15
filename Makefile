# Makefile

BUILDID=$(shell date +%Y%m%d-%H:%M:%S)

.PHONY: push pull

push:
	git add -A Makefile README.md src/
	git commit -m 'Automatic commit of successful build $(BUILDID)'
	git push origin master

pull:
	git pull https://github.com/jxunix/ear_project

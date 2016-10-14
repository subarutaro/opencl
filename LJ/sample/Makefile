all:
	make -C ./src
	cp ./src/md.out .
cuda:
	make -C ./src -f Makefile.cuda
	cp ./src/md.out .
clean:
	make clean -C ./src
	rm md.out

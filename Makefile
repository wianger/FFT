CC	= g++
CFLAGS	= -std=c++11

all: ./build/main

./build/main: main.cpp
	$(CC) $(CFLAGS) -o $@ $<

.PHONY: clean

clean: 
	rm -f ./build/*

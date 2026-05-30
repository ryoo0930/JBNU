CC      = gcc
WINCC   = x86_64-w64-mingw32-gcc
CFLAGS  = -Wall -O2 -Iinclude
SRCS    = src/main.c src/training.c src/test.c

all: main

main: $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o main -lm

win: $(SRCS)
	$(WINCC) $(CFLAGS) $(SRCS) -o main.exe -lm

clean:
	rm -f main main.exe

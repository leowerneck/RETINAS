CC       = gcc
CFLAGS  += -Wall -march=native -O2 -fopenmp -std=c99 -I../cross_correlation_c
LDFLAGS += -lm

SRC := $(wildcard *.c)
OBJ := $(SRC:.c=.o)

all: $(OBJ)

$(OBJ): %.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ)

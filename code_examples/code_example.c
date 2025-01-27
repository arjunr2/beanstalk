#define N 20000
// Global State
long fibonacci[2] = {1, 0};
int num_ct = 1;
volatile int p = 0;

void *fib_thread(void *tid) {
  int tnum = *(int *)tid;
  while (num_ct < N) {
  	p += tnum;
    long f1 = fibonacci[1];
    fibonacci[1] = fibonacci[0];
    fibonacci[0] += f1;
    num_ct++;
  }
  return NULL;
}

int main() {
  // Spawn + join 2 threads
  // invoking `fib_thread(tid)`
  spawn_threads(2, fib_thread);
	return 0;
}


#include "../vbuf.h"
#include "test_interface.h"
#include <dlfcn.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: %s <vbuf_file> <test_plugin.so>\n", argv[0]);
    return 1;
  }

  // 1. DATEI EINMALIG LADEN (mmap)
  vbuf_instance_t *inst = vbuf_open(argv[1]);
  if (!inst) {
    fprintf(stderr, "Fehler: Konnte Datei %s nicht öffnen.\n", argv[1]);
    return 1;
  }

  // 2. TEST-PLUGIN DYNAMISCH LADEN
  // Wir nutzen argv[2], da dies der Pfad zur .so Datei ist
  void *handle = dlopen(argv[2], RTLD_NOW); 
  if (!handle) {
    fprintf(stderr, "Fehler: Konnte %s nicht laden: %s\n", argv[2], dlerror());
    vbuf_close(inst);
    return 1;
  }

  vbuf_test_func func = (vbuf_test_func)dlsym(handle, "run_test");
  if (!func) {
    fprintf(stderr, "Fehler: Symbol 'run_test' in %s nicht gefunden!\n", argv[2]);
    dlclose(handle);
    vbuf_close(inst);
    return 1;
  }

// 3. EXECUTION
  func(inst); 

  // 4. CLEANUP
  vbuf_close(inst); 
  
  // Dann erst das Plugin entladen
  dlclose(handle); 

  return 0;
}

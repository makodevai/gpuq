#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>


void record_dl_error(const char** dl_error_buffer, size_t* dl_error_len, int append) {
    const char* err = dlerror();
    if (!err) {
        if (append)
            return;

        if (*dl_error_buffer) {
            free((void*)(*dl_error_buffer));
            (*dl_error_buffer) = NULL;
            (*dl_error_len) = 0;
        }

        return;
    }

    size_t err_len = strlen(err);

    if (!append && (*dl_error_buffer)) {
        free((void*)(*dl_error_buffer));
        (*dl_error_buffer) = NULL;
        (*dl_error_len) = 0;
    }

    size_t new_len = err_len + 3;
    if (*dl_error_len)
        new_len += (*dl_error_len) + 1;

    char* new_buffer = (char*)malloc(new_len + 1); // +1 for the null-terminator
    if (*dl_error_buffer) {
        strncpy(new_buffer, *dl_error_buffer, *dl_error_len);
        new_buffer[*dl_error_len] = '\n';
        (*dl_error_buffer) = new_buffer;
        new_buffer += (*dl_error_len) + 1;
    } else {
        (*dl_error_buffer) = new_buffer;
    }
    strncpy(new_buffer, " * ", 3);
    strncpy(new_buffer + 3, err, err_len);
    new_buffer[err_len + 3] = '\0';
    (*dl_error_len) = new_len;
}

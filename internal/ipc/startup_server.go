package ipc

import (
	"net/http"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/pipeline"
)

func StartServer(pipeline *pipeline.Pipeline) {

	http.HandleFunc("/initialize_loading", func(w http.ResponseWriter, r *http.Request) {

		pipeline.Initiate()
	})

	http.ListenAndServe(":8080", nil)
}

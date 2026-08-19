package com.eugen.vbufchat;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private final boolean remote = !BuildConfig.VBUF_REMOTE_URL.isEmpty();
    private final ExecutorService worker = Executors.newSingleThreadExecutor(runnable ->
            new Thread(null, runnable, "vbuf-worker", 8L * 1024L * 1024L));
    private TextView status;
    private TextView output;
    private EditText prompt;
    private Button send;

    @Override
    public void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(R.layout.activity_main);
         status = findViewById(R.id.status);
        output = findViewById(R.id.output);
        prompt = findViewById(R.id.prompt);
        send = findViewById(R.id.send);
        send.setEnabled(false);
        send.setOnClickListener(this::generate);
        openModel();
    }

    private File modelFile() {
        String name = remote ? "Qwen3-0.6B-Q8_0.semantic.vbuf" : "Qwen3-0.6B-Q8_0.vbuf";
        return new File(new File(getFilesDir(), "models"), name);
    }

    private void openModel() {
        File model = modelFile();
        worker.execute(() -> {
            if (remote) {
                String probe = NativeInference.probeRemote(model.getAbsolutePath(), BuildConfig.VBUF_REMOTE_URL);
                if (!probe.startsWith("PROBE_OK")) {
                    runOnUiThread(() -> status.setText(probe));
                    return;
                }
            }
            long start = System.nanoTime();
            String result = NativeInference.open(model.getAbsolutePath(), BuildConfig.VBUF_REMOTE_URL);
            long elapsedMs = (System.nanoTime() - start) / 1_000_000L;
            runOnUiThread(() -> {
                status.setText(result + " load_ms=" + elapsedMs);
                send.setEnabled(result.startsWith("OPEN_OK"));
            });
        });
    }

    private void generate(View ignored) {
        String text = prompt.getText().toString().trim();
        if (text.isEmpty()) return;
        send.setEnabled(false);
        status.setText("Generating...");
        worker.execute(() -> {
            long start = System.nanoTime();
            String result = NativeInference.generate(text, 16);
            long elapsedMs = (System.nanoTime() - start) / 1_000_000L;
            runOnUiThread(() -> {
                output.setText(result);
                status.setText("Generated in " + elapsedMs + " ms");
                send.setEnabled(true);
            });
        });
    }

    @Override
    protected void onDestroy() {
        worker.shutdownNow();
        NativeInference.closeModel();
        super.onDestroy();
    }
}

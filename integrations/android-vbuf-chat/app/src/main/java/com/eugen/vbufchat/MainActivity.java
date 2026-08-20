package com.eugen.vbufchat;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private final ExecutorService worker = Executors.newSingleThreadExecutor(runnable ->
            new Thread(null, runnable, "vbuf-worker", 8L * 1024L * 1024L));
    private TextView status;
    private TextView progress;
    private TextView output;
    private TextView metrics;
    private EditText prompt;
    private Button open;
    private Button generate;
    private Button cancel;
    private final long appStartMs = SystemClock.elapsedRealtime();
    private long appStartToReadyMs = -1L;
    private final Handler progressHandler = new Handler(Looper.getMainLooper());
    private final Runnable progressPoll = new Runnable() {
        @Override
        public void run() {
            final String value = NativeInference.progress();
            if (applyProgress(value)) progressHandler.postDelayed(this, 750L);
        }
    };

    private boolean applyProgress(String value) {
        final int newline = value.indexOf('\n');
        if (newline < 0) return false;
        final String header = value.substring(0, newline);
        progress.setText(header);
        final String text = value.substring(newline + 1);
        if (!text.isEmpty()) output.setText(text);
        if (header.contains("phase=PREFILL")) {
            status.setText("Prefill · batched prompt rows");
        } else if (header.contains("phase=DECODE")) {
            status.setText("Decode · autoregressive");
        }
        return value.startsWith("ACTIVE");
    }

    @Override
    public void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(R.layout.activity_main);
        status = findViewById(R.id.status);
        progress = findViewById(R.id.progress);
        output = findViewById(R.id.output);
        metrics = findViewById(R.id.metrics);
        prompt = findViewById(R.id.prompt);
        open = findViewById(R.id.open);
        generate = findViewById(R.id.generate);
        cancel = findViewById(R.id.cancel);
        open.setOnClickListener(ignored -> openModel());
        generate.setOnClickListener(this::generateText);
        cancel.setOnClickListener(ignored -> cancelGeneration());
        generate.setEnabled(false);
        cancel.setEnabled(false);
        status.setText("Idle · connect to the direct vBuf-ML runtime");
    }

    private File modelFile() {
        return new File(new File(getFilesDir(), "models"),
                "DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf");
    }

    private void openModel() {
        if (BuildConfig.VBUF_REMOTE_URL.isEmpty()) {
            status.setText("Failed · configure the HTTP payload endpoint");
            return;
        }
        open.setEnabled(false);
        status.setText("Opening model · semantic discovery");
        File model = modelFile();
        worker.execute(() -> {
            String result = NativeInference.open(model.getAbsolutePath(), BuildConfig.VBUF_REMOTE_URL, false);
            runOnUiThread(() -> {
                final boolean ready = result.startsWith("OPEN_OK");
                if (ready) appStartToReadyMs = SystemClock.elapsedRealtime() - appStartMs;
                status.setText(ready ? "Ready · direct vBuf-ML runtime" : result);
                generate.setEnabled(ready);
                open.setEnabled(!ready);
                refreshMetrics();
            });
        });
    }

    private void generateText(View ignored) {
        String text = prompt.getText().toString().trim();
        if (text.isEmpty()) {
            status.setText("Prompt is required");
            return;
        }
        generate.setEnabled(false);
        open.setEnabled(false);
        cancel.setEnabled(true);
        status.setText("Generating · bounded direct runtime");
        progressHandler.post(progressPoll);
        worker.execute(() -> {
            String result = NativeInference.generate(text, 4);
            final String finalProgress = NativeInference.progress();
            progressHandler.removeCallbacks(progressPoll);
            runOnUiThread(() -> {
                final boolean success = !result.startsWith("GEN_FAIL");
                applyProgress(finalProgress);
                output.setText(success ? result : "");
                status.setText(success ? "Completed" : result);
                generate.setEnabled(success);
                open.setEnabled(false);
                cancel.setEnabled(false);
                refreshMetrics();
            });
        });
    }

    private void cancelGeneration() {
        NativeInference.cancel();
        status.setText("Cancelling · waiting for the current dependency boundary");
        cancel.setEnabled(false);
    }

    private void refreshMetrics() {
        worker.execute(() -> {
            final String value = NativeInference.metrics();
            final String appMetrics = appStartToReadyMs >= 0
                    ? "APP_START_TO_READY_MS: " + appStartToReadyMs + "\n" : "";
            runOnUiThread(() -> metrics.setText(appMetrics + value));
        });
    }

    @Override
    protected void onDestroy() {
        progressHandler.removeCallbacks(progressPoll);
        NativeInference.cancel();
        worker.execute(NativeInference::closeModel);
        worker.shutdown();
        super.onDestroy();
    }
}

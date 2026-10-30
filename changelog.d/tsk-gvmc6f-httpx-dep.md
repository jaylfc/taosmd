httpx is now a declared core dependency. It was only installed through huggingface-hub, which switched to httpx2 in 2.2.0, so a clean install failed on `import taosmd`.

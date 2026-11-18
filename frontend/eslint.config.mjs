import js from '@eslint/js'
import tseslint from 'typescript-eslint'
import hooks from 'eslint-plugin-react-hooks'
export default tseslint.config(
  { ignores: ['dist/**', 'node_modules/**', 'test-results/**', 'playwright-report/**'] },
  js.configs.recommended, ...tseslint.configs.recommended,
  { files: ['**/*.{ts,tsx}'], plugins: { 'react-hooks': hooks }, languageOptions: { globals: {
    window: 'readonly', document: 'readonly', fetch: 'readonly', URL: 'readonly', File: 'readonly', Image: 'readonly',
    Blob: 'readonly', FormData: 'readonly', AbortController: 'readonly', AbortSignal: 'readonly', DOMException: 'readonly',
    performance: 'readonly', setTimeout: 'readonly', clearTimeout: 'readonly', process: 'readonly', console: 'readonly',
  } }, rules: { 'react-hooks/rules-of-hooks': 'error', 'react-hooks/exhaustive-deps': 'error', '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }] } },
)

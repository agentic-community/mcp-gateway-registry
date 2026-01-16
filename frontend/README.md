# MCP Gateway Registry Frontend

React-based frontend for the MCP Gateway Registry application.

## Development Setup

### Prerequisites

- Node.js 16+ and npm
- Backend server running on `http://localhost:7860` (configured in `package.json` proxy)

### Installation

```bash
npm install
```

Note: The postinstall script will automatically apply patches to dependencies.

### Running Development Server

```bash
npm start
```

The development server will start on `http://localhost:3000`.

## Important Configuration Notes

### webpack-dev-server v5 Compatibility Patch

This project uses `react-scripts` v5.0.1, which has a compatibility issue with `webpack-dev-server` v5. The project includes a patch to fix this issue.

**Problem**: react-scripts v5.0.1 uses deprecated webpack-dev-server hooks (`onBeforeSetupMiddleware` and `onAfterSetupMiddleware`) that were removed in webpack-dev-server v5.

**Solution**: We use `patch-package` to apply a patch that replaces the deprecated hooks with the modern `setupMiddlewares` API.

**Patch Location**: `patches/react-scripts+5.0.1.patch`

**How it Works**:
1. The patch modifies `node_modules/react-scripts/config/webpackDevServer.config.js`
2. Replaces deprecated hooks with `setupMiddlewares` function
3. The patch is automatically applied after `npm install` via the postinstall script

**If you encounter webpack-dev-server errors**:
1. Delete `node_modules` and `package-lock.json`
2. Run `npm install` to reinstall dependencies and reapply the patch
3. If the patch fails, check the `patches/react-scripts+5.0.1.patch` file for conflicts

## Available Scripts

- `npm start` - Start the development server
- `npm build` - Build the production bundle
- `npm test` - Run the test suite
- `npm run eject` - Eject from create-react-app (not recommended)

## Tech Stack

- React 18
- TypeScript
- Tailwind CSS
- React Router v6
- Heroicons
- Axios

## Project Structure

```
frontend/
├── src/
│   ├── components/    # Reusable React components
│   ├── contexts/      # React Context providers
│   ├── hooks/         # Custom React hooks
│   ├── pages/         # Page components
│   └── App.tsx        # Main application component
├── public/            # Static assets
├── patches/           # Dependency patches (managed by patch-package)
└── package.json
```

## Dependencies Management

### Using patch-package

This project uses `patch-package` to maintain patches for third-party dependencies. If you need to modify a dependency:

1. Make changes to files in `node_modules/`
2. Run `npx patch-package <package-name>`
3. Commit the generated patch file in `patches/` directory

The patches will be automatically applied after `npm install` via the postinstall script.

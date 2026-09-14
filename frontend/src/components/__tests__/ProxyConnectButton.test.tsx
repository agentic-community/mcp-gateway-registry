import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import ProxyConnectButton from '../ProxyConnectButton';

describe('ProxyConnectButton', () => {
  it('renders a disabled trigger, not nothing, when there is no client URL', () => {
    // An absent control cannot explain itself. A greyed, non-clickable Connect
    // states what would make it work, so "not proxied" is never a silent gap.
    render(<ProxyConnectButton clientUrl={null} />);
    const button = screen.getByRole('button', { name: /connect unavailable/i });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute('title', expect.stringMatching(/enable gateway routing/i));
    // Clicking a disabled trigger must never reveal the popover.
    fireEvent.click(button);
    expect(screen.queryByRole('dialog', { name: /connection details/i })).toBeNull();
  });

  it('shows a concise visible reason next to the label when disabled inline', () => {
    // The icon-only card control has no room for text, so it relies on the
    // tooltip; the labelled (modal row) variant has room and shows the reason.
    render(<ProxyConnectButton clientUrl={null} label="Connect" />);
    expect(screen.getByText(/enable gateway routing/i)).toBeInTheDocument();
  });

  it('enables the trigger and drops the disabled reason once proxied', () => {
    render(<ProxyConnectButton clientUrl="/gateway/skill/pdf" label="Connect" />);
    const button = screen.getByRole('button', { name: /show the client URL/i });
    expect(button).toBeEnabled();
    expect(screen.queryByText(/enable gateway routing/i)).toBeNull();

    fireEvent.click(button);
    // Trailing slash is deliberate: the nginx location ends in "/", so the bare
    // path 301s and a POST following that redirect gets downgraded to GET.
    expect(
      screen.getByText(`${window.location.origin}/gateway/skill/pdf/`),
    ).toBeInTheDocument();
  });

  it('opens a popover with the full same-origin client URL on click', () => {
    render(
      <ProxyConnectButton
        clientUrl="/gateway/rest-endpoint/abc"
        targetUrl="https://api.openai.com"
        connectNotes="Append /v1/chat/completions"
      />,
    );
    // Popover is closed until the button is clicked.
    expect(screen.queryByRole('dialog', { name: /connection details/i })).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: /connect/i }));

    expect(screen.getByRole('dialog', { name: /connection details/i })).toBeInTheDocument();
    // Full URL = origin + clientUrl + "/" (jsdom origin is http://localhost).
    expect(
      screen.getByText(`${window.location.origin}/gateway/rest-endpoint/abc/`),
    ).toBeInTheDocument();
    // Backend target, the operator-authored notes, and the fixed auth one-liner.
    expect(screen.getByText(/api\.openai\.com/)).toBeInTheDocument();
    expect(screen.getByText(/Append \/v1\/chat\/completions/)).toBeInTheDocument();
    expect(screen.getByText(/X-Authorization/)).toBeInTheDocument();
  });

  it('omits the notes paragraph when no connectNotes are provided', () => {
    render(<ProxyConnectButton clientUrl="/gateway/rest-endpoint/abc" />);
    fireEvent.click(screen.getByRole('button', { name: /connect/i }));
    // The fixed auth one-liner still shows; there is no operator-notes text.
    expect(screen.getByText(/X-Authorization/)).toBeInTheDocument();
    expect(screen.queryByText(/Append/)).toBeNull();
  });

  it('renders long notes as a wrap-safe, height-bounded block', () => {
    // Operator notes are free text up to 2000 chars and routinely contain an
    // unbroken URL or curl line far wider than the w-80 popover. Without a break
    // rule the text overflows the panel horizontally, and without a height cap the
    // popover grows past the card; both are layout-only, so the classes are the
    // contract jsdom can see.
    const notes =
      'Append /v1/forecast?latitude=39.2&longitude=-77.3&current=temperature_2m. ' +
      'Example: curl -s "http://localhost/gateway/rest-endpoint/abc/v1/forecast?latitude=39.2"';
    render(<ProxyConnectButton clientUrl="/gateway/rest-endpoint/abc" connectNotes={notes} />);
    fireEvent.click(screen.getByRole('button', { name: /connect/i }));

    const block = screen.getByText(notes);
    expect(block).toHaveClass('whitespace-pre-wrap', 'break-words');
    expect(block).toHaveClass('max-h-40', 'overflow-y-auto');
    // Full text stays reachable as a tooltip even while the block is scrolled.
    expect(block).toHaveAttribute('title', notes);
  });

  it('toggles the popover closed on a second click', () => {
    render(<ProxyConnectButton clientUrl="/gateway/rest-endpoint/abc" />);
    const button = screen.getByRole('button', { name: /connect/i });

    fireEvent.click(button);
    expect(screen.getByRole('dialog', { name: /connection details/i })).toBeInTheDocument();

    fireEvent.click(button);
    expect(screen.queryByRole('dialog', { name: /connection details/i })).toBeNull();
  });
});

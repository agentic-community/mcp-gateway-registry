import { describe, it, expect } from 'vitest';
import { screen } from '@testing-library/react';
import { render } from '@/test/utils';
import { TargetUserBanner } from '../TargetUserBanner';

describe('TargetUserBanner', () => {
  it('renders the target user email', () => {
    render(
      <TargetUserBanner
        targetUserEmail="test@example.com"
        targetUserId="test|user123"
      />
    );

    expect(screen.getByText(/Acting on user: test@example.com/)).toBeInTheDocument();
  });

  it('renders the target user ID', () => {
    render(
      <TargetUserBanner
        targetUserEmail="test@example.com"
        targetUserId="test|user123"
      />
    );

    expect(screen.getByText(/user_id: test\|user123/)).toBeInTheDocument();
  });

  it('shows admin action warning', () => {
    render(
      <TargetUserBanner
        targetUserEmail="admin@example.com"
        targetUserId="local|admin"
      />
    );

    expect(
      screen.getByText(/performing administrative actions on behalf of another user/)
    ).toBeInTheDocument();
  });

  it('shows audit logging notice', () => {
    render(
      <TargetUserBanner
        targetUserEmail="test@example.com"
        targetUserId="test|user123"
      />
    );

    expect(screen.getByText(/logged with your admin identity/)).toBeInTheDocument();
  });
});

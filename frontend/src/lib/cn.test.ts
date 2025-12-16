import { describe, it, expect } from 'vitest';
import { cn } from './cn';

describe('cn utility', () => {
  it('merges class names', () => {
    expect(cn('foo', 'bar')).toBe('foo bar');
  });

  it('handles conditional classes', () => {
    expect(cn('foo', true && 'bar', false && 'baz')).toBe('foo bar');
  });

  it('handles undefined and null', () => {
    expect(cn('foo', undefined, null, 'bar')).toBe('foo bar');
  });

  it('handles arrays', () => {
    expect(cn(['foo', 'bar'])).toBe('foo bar');
  });

  it('handles objects', () => {
    expect(cn({ foo: true, bar: false, baz: true })).toBe('foo baz');
  });

  it('merges Tailwind classes correctly', () => {
    // Later classes should override earlier ones
    expect(cn('px-2 py-1', 'px-4')).toBe('py-1 px-4');
  });

  it('merges conflicting Tailwind utilities', () => {
    expect(cn('text-red-500', 'text-blue-500')).toBe('text-blue-500');
  });

  it('preserves non-conflicting classes', () => {
    expect(cn('text-red-500', 'bg-blue-500')).toBe('text-red-500 bg-blue-500');
  });

  it('handles responsive variants', () => {
    expect(cn('md:text-red-500', 'md:text-blue-500')).toBe('md:text-blue-500');
  });

  it('handles complex combinations', () => {
    expect(
      cn(
        'base-class',
        true && 'conditional-class',
        { 'object-true': true, 'object-false': false },
        ['array-class'],
        undefined,
        null
      )
    ).toBe('base-class conditional-class object-true array-class');
  });

  it('handles empty input', () => {
    expect(cn()).toBe('');
  });
});

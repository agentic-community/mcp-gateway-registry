import React, { useState, useEffect, useRef } from 'react';
import { StarIcon } from '@heroicons/react/24/solid';
import { StarIcon as StarIconOutline } from '@heroicons/react/24/outline';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';

interface RatingDetail {
  user: string;
  rating: number;
}

interface StarRatingWidgetProps {
  resourceType: 'agents' | 'servers' | 'skills' | 'virtual-servers' | 'custom';
  path: string;
  initialRating?: number;
  initialCount?: number;
  // Full per-user rating list from the list payload. Supplying this lets the
  // widget render the current user's own rating without a per-card /rating
  // fetch (the N+1 this component used to cause across a page of cards).
  ratingDetails?: RatingDetail[];
  authToken?: string | null;
  onShowToast?: (message: string, type: 'success' | 'error') => void;
  onRatingUpdate?: (newRating: number) => void;
}


const StarRatingWidget: React.FC<StarRatingWidgetProps> = ({
  resourceType,
  path,
  initialRating = 0,
  initialCount = 0,
  ratingDetails,
  authToken,
  onShowToast,
  onRatingUpdate
}) => {
  const { user } = useAuth();
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [selectedRating, setSelectedRating] = useState<number | null>(null);
  const [hoverRating, setHoverRating] = useState<number | null>(null);
  const [currentUserRating, setCurrentUserRating] = useState<number | null>(null);
  const [averageRating, setAverageRating] = useState(initialRating);
  const [ratingCount, setRatingCount] = useState(initialCount);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const [dropdownPos, setDropdownPos] = useState<{ top: number; left: number }>({ top: 0, left: 0 });
  const dropdownRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  // Singular noun for labels (e.g. "servers" -> "server"). "custom" has no
  // trailing plural to strip, so it maps to a generic "entry".
  const singularLabel = resourceType === 'custom' ? 'entry' : resourceType.slice(0, -1);


  // Derive display + the current user's own rating from the list payload props.
  // No fetch needed: the list endpoint already returns num_stars (initialRating),
  // the rating count, and rating_details, so a page of cards costs zero extra
  // requests instead of one /rating call per card.
  useEffect(() => {
    setAverageRating(initialRating);
    setRatingCount(ratingDetails ? ratingDetails.length : initialCount);

    const username = user?.username;
    const ownRating = ratingDetails?.find((detail) => detail.user === username);
    if (ownRating) {
      setCurrentUserRating(ownRating.rating);
      setSelectedRating(ownRating.rating);
    } else {
      setCurrentUserRating(null);
      setSelectedRating(null);
    }
  }, [resourceType, path, initialRating, initialCount, ratingDetails, user?.username]);


  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    };

    if (isDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isDropdownOpen]);


  const handleSubmitRating = async () => {
    console.log('handleSubmitRating called', { selectedRating, authToken: !!authToken });
    if (!selectedRating) {
      console.log('Validation failed - no rating selected');
      return;
    }

    setIsSubmitting(true);
    try {
      // Build headers - use Bearer token if provided, otherwise rely on cookies
      const headers = authToken ? { Authorization: `Bearer ${authToken}` } : undefined;
      // Both servers and agents now use consistent path parameter pattern
      const url = `/api/${resourceType}${path}/rate`;
      console.log('Submitting rating to:', url, { rating: selectedRating });

      const response = await axios.post(
        url,
        { rating: selectedRating },
        headers ? { headers } : undefined
      );

      console.log('Rating response:', response.data);
      const newAverageRating = response.data.average_rating;
      setAverageRating(newAverageRating);
      setCurrentUserRating(selectedRating);

      // Update count (increment if new rating, keep same if update)
      if (!currentUserRating) {
        setRatingCount(prev => prev + 1);
      }

      setShowSuccess(true);

      if (onShowToast) {
        onShowToast(
          currentUserRating ? 'Rating updated successfully!' : 'Rating submitted successfully!',
          'success'
        );
      }

      if (onRatingUpdate) {
        onRatingUpdate(newAverageRating);
      }

      // Auto-close after 2 seconds
      console.log('Setting timeout to close dialog...');
      setTimeout(() => {
        console.log('Closing dialog now');
        setShowSuccess(false);
        setIsDropdownOpen(false);
      }, 2000);
    } catch (error: any) {
      console.error('Failed to submit rating:', error);
      console.error('Error details:', error.response?.data);
      if (onShowToast) {
        onShowToast(
          error.response?.data?.detail || 'Failed to submit rating',
          'error'
        );
      }
    } finally {
      setIsSubmitting(false);
    }
  };


  const handleStarClick = (rating: number) => {
    setSelectedRating(rating);
  };


  const handleCancel = () => {
    setIsDropdownOpen(false);
    setSelectedRating(currentUserRating);
    setHoverRating(null);
  };


  const renderStars = (count: number, filled: boolean, size: 'small' | 'large' = 'large') => {
    const sizeClass = size === 'small' ? 'h-4 w-4' : 'h-6 w-6';
    const IconComponent = filled ? StarIcon : StarIconOutline;

    return (
      <IconComponent
        className={`${sizeClass} ${filled ? 'text-warning-400' : 'text-gray-300 dark:text-gray-600'}`}
      />
    );
  };


  const displayRating = hoverRating !== null ? hoverRating : (selectedRating || currentUserRating || 0);


  return (
    <div className="relative" ref={dropdownRef}>
      {/* Rating Display - Clickable */}
      <button
        ref={buttonRef}
        onClick={() => {
          if (!isDropdownOpen && buttonRef.current) {
            const rect = buttonRef.current.getBoundingClientRect();
            setDropdownPos({ top: rect.bottom + 8, left: rect.left });
          }
          setIsDropdownOpen(!isDropdownOpen);
        }}
        className="flex items-center gap-2 hover:bg-warning-50 dark:hover:bg-warning-900/20 p-2 rounded-lg transition-colors duration-200"
        title={`Click to rate this ${singularLabel}`}
        aria-label={`Rate this ${singularLabel}`}
        aria-expanded={isDropdownOpen}
        aria-haspopup="dialog"
      >
        <div className="p-1.5 bg-warning-50 dark:bg-warning-900/30 rounded">
          <StarIcon className="h-4 w-4 text-warning-600 dark:text-warning-400" />
        </div>
        <div>
          <div className="text-sm font-semibold text-gray-900 dark:text-white">
            {averageRating > 0 ? averageRating.toFixed(1) : '0'}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">
            {ratingCount === 0 ? 'No ratings' : `${ratingCount} rating${ratingCount !== 1 ? 's' : ''}`}
          </div>
        </div>
      </button>

      {/* Rating Dropdown */}
      {isDropdownOpen && (
        <div
          className="fixed w-80 bg-surface-overlay rounded-lg shadow-xl border border-line p-4"
          style={{ top: dropdownPos.top, left: dropdownPos.left, zIndex: 9999 }}
          role="dialog"
          aria-label={`${singularLabel} rating form`}
        >
          {/* Success State */}
          {showSuccess ? (
            <div className="text-center py-6">
              <div className="inline-flex items-center justify-center w-12 h-12 bg-success-100 dark:bg-success-900/30 rounded-full mb-3">
                <svg className="w-6 h-6 text-success-600 dark:text-success-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
                Rating {currentUserRating && selectedRating !== currentUserRating ? 'updated' : 'submitted'}!
              </h4>
              <div className="flex justify-center items-center gap-1 mb-2">
                {[1, 2, 3, 4, 5].map((star) => (
                  <div key={star}>
                    {renderStars(star, star <= (selectedRating || 0), 'small')}
                  </div>
                ))}
                <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
                  ({selectedRating} stars)
                </span>
              </div>
              <p className="text-sm text-fg-muted">
                New average: {averageRating.toFixed(1)}{' '}
                <StarIcon className="inline h-4 w-4 -mt-0.5 text-warning-400" aria-hidden="true" />
              </p>
            </div>
          ) : isSubmitting ? (
            // Loading State
            <div className="text-center py-6">
              <div className="inline-flex items-center justify-center w-12 h-12 mb-3">
                <svg className="animate-spin h-8 w-8 text-primary-600" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              </div>
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                Submitting your rating...
              </p>
            </div>
          ) : (
            // Rating Form
            <>
              <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-1">
                {currentUserRating ? 'Update your rating:' : `Rate this ${singularLabel}:`}
              </h4>
              {currentUserRating && (
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                  Currently: {currentUserRating} stars
                </p>
              )}

              {/* Star Selection */}
              <div
                className="flex items-center justify-center gap-2 my-4"
                role="radiogroup"
                aria-label="Select rating"
              >
                {[1, 2, 3, 4, 5].map((star) => (
                  <button
                    key={star}
                    onClick={() => handleStarClick(star)}
                    onMouseEnter={() => setHoverRating(star)}
                    onMouseLeave={() => setHoverRating(null)}
                    className="p-1 hover:scale-110 transition-transform duration-150 rounded"
                    role="radio"
                    aria-checked={selectedRating === star}
                    aria-label={`${star} star${star !== 1 ? 's' : ''}`}
                  >
                    {renderStars(star, star <= displayRating)}
                  </button>
                ))}
              </div>

              {/* Rating Preview Text */}
              {displayRating > 0 && (
                <p className="text-center text-sm text-gray-600 dark:text-gray-400 mb-4">
                  {displayRating} star{displayRating !== 1 ? 's' : ''}
                </p>
              )}

              {/* Action Buttons */}
              <div className="flex gap-2 mt-4">
                <button
                  onClick={handleCancel}
                  className="btn-secondary flex-1"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSubmitRating}
                  disabled={!selectedRating}
                  className="btn-primary flex-1"
                >
                  {currentUserRating ? 'Update Rating' : 'Submit Rating'}
                  {selectedRating && (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};


export default StarRatingWidget;

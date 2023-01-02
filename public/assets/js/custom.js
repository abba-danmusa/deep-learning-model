jQuery(function ($) {
	'use strict';

    // Start Menu
    $(window).on('scroll', function() {
        if ($(this).scrollTop() >50) {
            $('.navbar-light').addClass('menu-shrink');
        } else {
            $('.navbar-light').removeClass('menu-shrink');
        }
    });			
    $('.navbar-nav .nav-item .nav-link').on('click', function(e){
        var anchor = $(this);
        $('html, body').stop().animate({
            scrollTop: $(anchor.attr('href')).offset().top - 50
        }, 50);
        e.preventDefault();
    });
    $(document).on('click','.navbar-collapse.show',function(e) {
        if( $(e.target).is('a') && $(e.target).attr('class') != 'dropdown-toggle' ) {
            $(this).collapse('hide');
        }
	});	
	$('.one-help-area .help-item .advisor-link a').on('click', function(e){
        var anchor = $(this);
        $('html, body').stop().animate({
            scrollTop: $(anchor.attr('href')).offset().top - 50
        }, 1500);
        e.preventDefault();
    });	

	// Sorting Portfolio 
	try {
        var mixer = mixitup('.shorting', {
            controls: {
                toggleDefault: 'none'
            }
        });
    } catch (err) {}

	// Banner Slider 
	$('.banner-slider').owlCarousel({
		items: 1,
		loop: true,
		margin: 15,
		singleItem: true,
		nav: true,
		dots: false,
		smartSpeed: 1000,
		autoplay: true,
		autoplayTimeout: 4000,
		autoplayHoverPause: true,
		navText: [
            "<i class='bx bx-left-arrow-alt'></i>",
            "<i class='bx bx-right-arrow-alt'></i>"
        ],
    });

	// Among Slider 
	$('.among-slider').owlCarousel({
		items: 3,
		loop: true,
		margin: 0,
		nav: false,
		dots: true,
		smartSpeed: 1000,
		autoplay: true,
		autoplayTimeout: 4000,
		autoplayHoverPause: true,
		center: true,
		responsive:{
			0:{
				items:1,
			},
			600:{
				items:2,
			},
			1000:{
				items:3,
			}
		}
	});

	// Popup Youtube 
	$('.popup-youtube').magnificPopup({
		disableOn: 320,
		type: 'iframe',
		mainClass: 'mfp-fade',
		removalDelay: 160,
		preloader: false,
		fixedContentPos: false
	});

	// Nice Select
	$('select').niceSelect();

	// Accordion 
	$('.accordion > li:eq(0) a').addClass('active').next().slideDown();
	$('.accordion a').click(function(j) {
		var dropDown = $(this).closest('li').find('p');
		$(this).closest('.accordion').find('p').not(dropDown).slideUp();
		if ($(this).hasClass('active')) {
			$(this).removeClass('active');
		} else {
			$(this).closest('.accordion').find('a.active').removeClass('active');
			$(this).addClass('active');
		}
		dropDown.stop(false, true).slideToggle();
		j.preventDefault();
	});

	// Banner Slider Two 
	$('.two-banner-slider').owlCarousel({
		items: 1,
		loop: true,
		margin: 15,
		singleItem: true,
		nav: true,
		dots: false,
		smartSpeed: 1000,
		autoplay: true,
		autoplayTimeout: 4000,
		autoplayHoverPause: true,
		navText: [
            "<i class='bx bx-left-arrow-alt'></i>",
            "<i class='bx bx-right-arrow-alt'></i>"
        ],
	});

	// Review Slider Two 
	$('.three-review-slider').owlCarousel({
		items: 1,
		loop: true,
		margin: 15,
		singleItem: true,
		nav: false,
		dots: true,
		smartSpeed: 1000,
		autoplay: true,
		autoplayTimeout: 4000,
		autoplayHoverPause: true,
	});

	// Step Slider 
	$('.four-step-slider').owlCarousel({
		center: true,
		items: 4,
		loop: true,
		margin: 50,
		nav: false,
		dots: false,
		smartSpeed: 1000,
		autoplay: true,
		autoplayTimeout: 3000,
		autoplayHoverPause: true,
		responsive:{
			0:{
				items:1,
			},
			600:{
				items:2,
			},
			1000:{
				items:4,
			}
		}
	});

	// Step Slider Two 
	$('.four-step-slider-two').owlCarousel({
		center: true,
		items: 4,
		loop: true,
		margin: 50,
		nav: false,
		dots: true,
		smartSpeed: 1000,
		autoplay: true,
		autoplayTimeout: 3000,
		autoplayHoverPause: true,
		responsive:{
			0:{
				items:1,
			},
			600:{
				items:2,
			},
			1000:{
				items:4,
			}
		}
	});

	// Loan Slider 
	$('.five-loan-slider').owlCarousel({
		items: 4,
		loop: true,
		margin: 15,
		nav: false,
		dots: true,
		smartSpeed: 1000,
		autoplay: true,
		autoplayTimeout: 4000,
		autoplayHoverPause: true,
		responsive:{
			0:{
				items:1,
			},
			600:{
				items:2,
			},
			1000:{
				items:4,
			}
		}
	});

	// Lover Slider 
	$('.six-lover-slider').owlCarousel({
		center: true,
		items: 4,
		loop: true,
		margin: 25,
		nav: false,
		dots: true,
		smartSpeed: 1000,
		autoplay: true,
		autoplayTimeout: 4000,
		autoplayHoverPause: true,
		responsive:{
			0:{
				items:1,
			},
			600:{
				items:2,
			},
			1000:{
				items:4,
			}
		}
	});

	// Subscribe Form 
	$(".newsletter-form").validator().on("submit", function (event) {
		if (event.isDefaultPrevented()) {
			// Hande the invalid form...
			formErrorSub();
			submitMSGSub(false, "Please enter your email correctly.");
		} else {
			// Everything looks good!
			event.preventDefault();
		}
	});
	function callbackFunction (resp) {
		if (resp.result === "success") {
			formSuccessSub();
		}
		else {
			formErrorSub();
		}
	}
	function formSuccessSub(){
		$(".newsletter-form")[0].reset();
		submitMSGSub(true, "Thank you for subscribing!");
		setTimeout(function() {
			$("#validator-newsletter").addClass('hide');
		}, 4000)
	}
	function formErrorSub(){
		$(".newsletter-form").addClass("animated shake");
		setTimeout(function() {
			$(".newsletter-form").removeClass("animated shake");
		}, 1000)
	}
	function submitMSGSub(valid, msg){
		if(valid){
			var msgClasses = "validation-success";
		} else {
			var msgClasses = "validation-danger";
		}
		$("#validator-newsletter").removeClass().addClass(msgClasses).text(msg);
	}
	
	// AJAX Mail Chimp 
	$(".newsletter-form").ajaxChimp({
		url: "https://envytheme.us20.list-manage.com/subscribe/post?u=60e1ffe2e8a68ce1204cd39a5&amp;id=42d6d188d9", // Your url MailChimp
		callback: callbackFunction
	});

	// Preloader 
	jQuery(window).on('load',function(){
		jQuery(".loader").fadeOut(500);
	});

	// Back to Top 
	$('body').append('<div id="toTop" class="back-to-top-btn"><i class="bx bxs-up-arrow-alt"></i></div>');
	$(window).scroll(function () {
		if ($(this).scrollTop() != 0) {
			$('#toTop').fadeIn();
		} else {
			$('#toTop').fadeOut();
		}
	}); 
	$('#toTop').on('click', function(){
		$("html, body").animate({ scrollTop: 0 }, 900);
		return false;
	});

	/*
	New Demo JSS
	======================================================*/ 

	// Header Sticky
	$(window).on('scroll',function() {
		if ($(this).scrollTop() > 120){  
			$('.navbar').addClass("is-sticky");
		}
		else{
			$('.navbar').removeClass("is-sticky");
		}
	});

	// Why Choose US Slider 
	$('.why-choose-us-slider').owlCarousel({
		loop: true,
		margin: 25,
		nav: false,
		dots: true,

		responsive:{
			0:{
				items: 1,
			},
			576:{
				items: 1,
			},
			768:{
				items: 2,
			},
			1024:{
				items: 3,
			},
			1200:{
				items: 3,
			}
		}
	});

	// Staff Slider 
	$('.staff-slider').owlCarousel({
		loop: true,
		margin: 25,
		nav: false,
		dots: true,

		responsive:{
			0:{
				items: 1,
			},
			576:{
				items: 2,
			},
			768:{
				items: 2,
			},
			1024:{
				items: 3,
			},
			1200:{
				items: 4,
			}
		}
	});

	// Testimonials Style Slider 
	$('.testimonials-style-slider').owlCarousel({
		loop: true,
		margin: 25,
		nav: false,
		dots: true,
		items: 1,
	});

	// Offer Style Slider 
	$('.offer-style-slider').owlCarousel({
		loop: true,
		margin: 25,
		nav: false,
		dots: true,

		responsive:{
			0:{
				items: 1,
			},
			576:{
				items: 1,
			},
			768:{
				items: 2,
			},
			1024:{
				items: 3,
			},
			1200:{
				items: 3,
			}
		}
	});

	// Beyond Slider JS
	$(document).ready(function() {
		var bigimage = $("#beyond-style-item");
		var thumbs = $("#thumbs");
		//var totalslides = 10;
		var syncedSecondary = true;
	  
		bigimage
		  .owlCarousel({
		  items: 1,
		  slideSpeed: 2000,
		  nav: true,
		  autoplay: true,
		  dots: false,
		  nav: false,
		  loop: true,
		  responsiveRefreshRate: 200,
		  navText: [
				"<i class='bx bx-left-arrow-alt'></i>",
				"<i class='bx bx-right-arrow-alt'></i>"
			],
		})
		.on("changed.owl.carousel", syncPosition);
	  
		thumbs
		  .on("initialized.owl.carousel", function() {
		  thumbs
			.find(".owl-item")
			.eq(0)
			.addClass("current");
		})
		.owlCarousel({
			loop: true,
			dots: false,
			nav: true,
			autoplay: false,
			active: true,
			navText: [
				"<i class='bx bx-left-arrow-alt'></i>",
				"<i class='bx bx-right-arrow-alt'></i>"
			],
			smartSpeed: 200,
			slideSpeed: 500,
			slideBy: 8,
			responsiveRefreshRate: 100,
			responsive: {
				0: {
					items: 2
				},
				1024: {
					items: 4
				},
				1200: {
					items: 4
				}
			}
		})
		.on("changed.owl.carousel", syncPosition2);
	  
		function syncPosition(el) {
		//if loop is set to false, then you have to uncomment the next line
		//var current = el.item.index;
	  
		//to disable loop, comment this block
		  var count = el.item.count - 1;
		  var current = Math.round(el.item.index - el.item.count / 2 - 0.5);
	  
		  if (current < 0) {
			current = count;
		  }
		  if (current > count) {
			current = 0;
		  }
		  //to this
		  thumbs
			.find(".owl-item")
			.removeClass("current")
			.eq(current)
			.addClass("current");
		  var onscreen = thumbs.find(".owl-item.active").length - 1;
		  var start = thumbs
		  .find(".owl-item.active")
		  .first()
		  .index();
		  var end = thumbs
		  .find(".owl-item.active")
		  .last()
		  .index();
	  
		  if (current > end) {
			thumbs.data("owl.carousel").to(current, 100, true);
		  }
		  if (current < start) {
			thumbs.data("owl.carousel").to(current - onscreen, 100, true);
		  }
		}
	  
		function syncPosition2(el) {
		  if (syncedSecondary) {
			var number = el.item.index;
			bigimage.data("owl.carousel").to(number, 100, true);
		  }
		}
	  
		thumbs.on("click", ".owl-item", function(e) {
		  e.preventDefault();
		  var number = $(this).index();
		  bigimage.data("owl.carousel").to(number, 300, true);
		});
	});

	// Find Own Loan Slider  JS
	$('.find-own-loan-slider').owlCarousel({
		loop: true,
		margin: 25,
		nav: true,
		dots: false,
		navText: [
			"<i class='bx bx-left-arrow-alt'></i>",
			"<i class='bx bx-right-arrow-alt'></i>"
		],
		responsive:{
			0:{
				items: 1,
			},
			576:{
				items: 2,
			},
			768:{
				items: 2,
			},
			1024:{
				items: 3,
			},
			1200:{
				items: 4,
			}
		}
	});

	// Blog Style Slider JS
	$('.blog-style-slider').owlCarousel({
		loop: true,
		margin: 25,
		nav: false,
		dots: true,

		responsive:{
			0:{
				items: 1,
			},
			576:{
				items: 2,
			},
			768:{
				items: 2,
			},
			1024:{
				items: 3,
			},
			1200:{
				items: 3,
			}
		}
	});

	// Buy Now Btn
    $('body').append("<a href='https://themeforest.net/checkout/from_item/27513076?license=regular&support=bundle_6month&_ga=2.45883619.634514020.1646539215-1425290503.1590986634' class='buy-now-btn' target='_blank'><img src='assets/img/envato.png' alt='envato'/>Buy Now</a>");

}(jQuery));

// function to set a given theme/color-scheme
function setTheme(themeName) {
	localStorage.setItem('pento_theme', themeName);
	document.documentElement.className = themeName;
}
// function to toggle between light and dark theme
function toggleTheme() {
	if (localStorage.getItem('pento_theme') === 'theme-dark') {
		setTheme('theme-light');
	} else {
		setTheme('theme-dark');
	}
}
// Immediately invoked function to set the theme on initial load
(function () {
	if (localStorage.getItem('pento_theme') === 'theme-dark') {
		setTheme('theme-dark');
		document.getElementById('slider').checked = false;
	} else {
		setTheme('theme-light');
	  document.getElementById('slider').checked = true;
	}
})();